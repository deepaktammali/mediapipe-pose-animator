/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/

/* eslint-disable max-len */

import * as posenetModule from '@tensorflow-models/posenet';
import * as faceMeshModule from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs';
import * as paper from 'paper';
import dat from 'dat.gui';
import Stats from 'stats.js';
import {Holistic} from '@mediapipe/holistic';

import {drawKeypoints, drawPoint, drawSkeleton, isMobile, toggleLoadingUI, setStatusText} from './utils/demoUtils';
import {SVGUtils} from './utils/svgUtils';
import {PoseIllustration} from './illustrationGen/illustration';
import {Skeleton, facePartName2Index} from './illustrationGen/skeleton';
import {FileUtils} from './utils/fileUtils';

import * as girlSVG from './resources/illustration/girl.svg';
import * as boySVG from './resources/illustration/boy.svg';
import * as abstractSVG from './resources/illustration/abstract.svg';
import * as blathersSVG from './resources/illustration/blathers.svg';
import * as tomNookSVG from './resources/illustration/tom-nook.svg';
import {Camera} from '@mediapipe/camera_utils';

const mediaPosePartToIndexMap = {
  'nose': 0,
  'leftEye': 2,
  'rightEye': 5,
  'rightEar': 8,
  'leftEar': 7,
  'rightShoulder': 12,
  'leftShoulder': 11,
  'rightElbow': 14,
  'leftElbow': 13,
  'rightWrist': 16,
  'leftWrist': 15,
  'rightHip': 24,
  'leftHip': 23,
  'rightKnee': 26,
  'leftKnee': 25,
  'rightAnkle': 28,
  'leftAnkle': 27,
};

// Camera stream video element
let video;
let videoWidth = 300;
let videoHeight = 300;

// Canvas
let faceDetection = null;
let illustration = null;
let canvasScope;
let canvasWidth = 800;
let canvasHeight = 800;

// ML models
// let facemesh;
let posenet;
let minPoseConfidence = 0.15;
let minPartConfidence = 0.1;
// let nmsRadius = 30.0;

// Misc
let mobile = false;
const stats = new Stats();

const avatarSvgs = {
  'girl': girlSVG,
  'boy': boySVG,
  'abstract': abstractSVG,
  'blathers': blathersSVG,
  'tom-nook': tomNookSVG,
};

const holistic = new Holistic({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
  },
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: true,
  smoothSegmentation: true,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});


  async function setupCamera() {
    let videoElement = document.getElementById('video');

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await holistic.send({image: videoElement});
      },
      width: 1280,
      height: 720,
    });

    camera.start();

    return new Promise((resolve) => {
        console.log(videoElement);
        videoElement.addEventListener('loadedmetadata', ()=>{
          resolve(videoElement);
        });
      });
  }

  async function loadVideo() {
    const videoElement = await setupCamera();
    videoElement.play();

    return videoElement;
  }

const defaultPoseNetArchitecture = 'MobileNetV1';
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

const guiState = {
  avatarSVG: Object.keys(avatarSvgs)[0],
  debug: {
    showDetectionDebug: true,
    showIllustrationDebug: false,
  },
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras) {
  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  let multi = gui.addFolder('Image');
  gui.add(guiState, 'avatarSVG', Object.keys(avatarSvgs)).onChange(() => parseSVG(avatarSvgs[guiState.avatarSVG]));
  multi.open();

  let output = gui.addFolder('Debug control');
  output.add(guiState.debug, 'showDetectionDebug');
  output.add(guiState.debug, 'showIllustrationDebug');
  output.open();
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
const detectPoseInRealTime = (video, results) => {
  const {image} = results;

  if (!results.faceLandmarks || !results.poseLandmarks) {
    return;
  }

  const canvas = document.getElementById('output');
  const keypointCanvas = document.getElementById('keypoints');
  const videoCtx = canvas.getContext('2d');
  const keypointCtx = keypointCanvas.getContext('2d');

  canvas.width = image.width/4;
  canvas.height = image.height/4;
  keypointCanvas.width = image.width/4;
  keypointCanvas.height = image.height/4;

  // async function poseDetectionFrame() {
  // Begin monitoring code for frames per second
  stats.begin();

  let poses = [];

  videoCtx.clearRect(0, 0, image.width/4, image.height/4);
  // Draw video
  videoCtx.save();
  videoCtx.scale(-1, 1);
  videoCtx.translate(-image.width/4, 0);
  videoCtx.drawImage(video, 0, 0, image.width/4, image.height/4);
  videoCtx.restore();

  // Creates a tensor from an image
  const input = tf.browser.fromPixels(canvas);
  let faceLandmarksScaled = results.faceLandmarks.map((kp) => {
    return [kp.x * image.width, kp.y * image.height, kp.z];
  });

  faceDetection = [{
    'faceInViewConfidence': 1,
    'scaledMesh': faceLandmarksScaled,
  }];

  let poseLandmarksScaled = Object.keys(mediaPosePartToIndexMap).map((partName) => {
    let kp = results.poseLandmarks[mediaPosePartToIndexMap[partName]];
    return {
      'part': partName,
      'score': kp.visibility,
      'position': {
        'x': kp.x * image.width,
        'y': kp.y * image.height,
        'z': kp.z,
      },
    };
  });

  poses = [{
    'score': 1,
    'keypoints': poseLandmarksScaled,
  }];

  input.dispose();

  keypointCtx.clearRect(0, 0, image.width/4, image.height/4);
  if (guiState.debug.showDetectionDebug) {
    poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
        drawKeypoints(keypoints, minPartConfidence, keypointCtx);
        drawSkeleton(keypoints, minPartConfidence, keypointCtx);
      }
    });

    faceDetection.forEach((face) => {
      Object.values(facePartName2Index).forEach((index) => {
        let p = face.scaledMesh[index];
        drawPoint(keypointCtx, p[1], p[0], 2, 'red');
      });
    });
  }

  canvasScope.project.clear();

  if (poses.length >= 1 && illustration) {
    // Skeleton.flipPose(poses[0]);

    if (faceDetection && faceDetection.length > 0) {
      let face = Skeleton.toFaceFrame(faceDetection[0]);
      illustration.updateSkeleton(poses[0], face);
    } else {
      illustration.updateSkeleton(poses[0], null);
    }
    illustration.draw(canvasScope, image.width, image.height);

    if (guiState.debug.showIllustrationDebug) {
      illustration.debugDraw(canvasScope);
    }
  }

  canvasScope.project.activeLayer.scale(
    canvasWidth / image.width,
    canvasHeight / image.height,
    new canvasScope.Point(0, 0));

  // End monitoring code for frames per second
  stats.end();
};

function setupCanvas() {
  mobile = isMobile();
  if (mobile) {
    canvasWidth = Math.min(window.innerWidth, window.innerHeight);
    canvasHeight = canvasWidth;
    videoWidth *= 0.7;
    videoHeight *= 0.7;
  }

  canvasScope = paper;
  let canvas = document.querySelector('.illustration-canvas'); ;
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  canvasScope.setup(canvas);
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  setupCanvas();

  toggleLoadingUI(true);
  setStatusText('Loading PoseNet model...');
  posenet = await posenetModule.load({
    architecture: defaultPoseNetArchitecture,
    outputStride: defaultStride,
    inputResolution: defaultInputResolution,
    multiplier: defaultMultiplier,
    quantBytes: defaultQuantBytes,
  });
  setStatusText('Loading FaceMesh model...');
  facemesh = await faceMeshModule.load();

  setStatusText('Loading Avatar file...');
  await parseSVG(Object.values(avatarSvgs)[0]);

  setStatusText('Setting up camera...');

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this device type is not supported yet, ' +
      'or this browser does not support video capture: ' + e.toString();
    info.style.display = 'block';
    throw e;
  }

  holistic.onResults((results) => detectPoseInRealTime(video, results));

  setupGui([], posenet);
  setupFPS();

  toggleLoadingUI(false);
  // detectPoseInRealTime(video, posenet);
}

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
FileUtils.setDragDropHandler((result) => {
  parseSVG(result);
});

async function parseSVG(target) {
  let svgScope = await SVGUtils.importSVG(target /* SVG string or file path */);
  let skeleton = new Skeleton(svgScope);
  illustration = new PoseIllustration(canvasScope);
  illustration.bindSkeleton(skeleton, svgScope);
}

bindPage();
