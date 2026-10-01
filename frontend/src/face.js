import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM);
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
}

const mean = (a, b) => (a + b) / 2;

// Converts a MediaPipe result into the ~10 numbers the backend needs.
export function extractFeatures(result) {
  if (!result.faceLandmarks?.length) return { face_found: false };

  const bs = {};
  for (const c of result.faceBlendshapes?.[0]?.categories ?? []) bs[c.categoryName] = c.score;

  // Head pose from the 4x4 column-major transformation matrix.
  let yaw = 0, pitch = 0;
  const m = result.facialTransformationMatrixes?.[0]?.data;
  if (m) {
    const r02 = m[8], r12 = m[9], r22 = m[10];
    yaw = (Math.atan2(r02, r22) * 180) / Math.PI;
    pitch = (Math.asin(Math.max(-1, Math.min(1, -r12))) * 180) / Math.PI;
  }

  return {
    face_found: true,
    blink_left: bs.eyeBlinkLeft ?? 0,
    blink_right: bs.eyeBlinkRight ?? 0,
    smile: mean(bs.mouthSmileLeft ?? 0, bs.mouthSmileRight ?? 0),
    brow_down: mean(bs.browDownLeft ?? 0, bs.browDownRight ?? 0),
    brow_up: bs.browInnerUp ?? 0,
    jaw_open: bs.jawOpen ?? 0,
    yaw,
    pitch,
  };
}
