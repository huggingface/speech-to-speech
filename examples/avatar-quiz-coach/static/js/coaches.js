// @ts-check
/**
 * The coaches: who they are, how they sound, and which renderer draws them.
 *
 * - Max: TalkingHead (three.js) with the AvatarSDK sample avatar, upper body
 *   with hand gestures (free for non-commercial use).
 * - Ada: a photoreal Gaussian-splat head ("Nyx", from LAM output, MIT, via
 *   @myned-ai/avatar-chat-widget) drawn by @myned-ai/gsplat-flame-avatar-renderer
 *   and driven with ARKit blendshapes; head and shoulders only.
 * - Reachy Mini, Reachy 2 and Microduck: Pollen Robotics' robots (Apache-2.0
 *   models), drawn by robot-avatar.js. Reachy Mini and Reachy 2 talk with
 *   their heads and antennas, Microduck with its beak; all react with
 *   emotions recorded on the real Reachy Mini.
 *
 * `lipGain` / `jawGain` say how strongly a rig's mouth shapes read.
 */
export const COACHES = {
  max: {
    name: "Max",
    voice: "en-arthur_m",
    renderer: "talkinghead",
    lipGain: 1.35,
    jawGain: 1,
    avatar: {
      url: "static/avatars/avatarsdk.glb",
      body: "M",
      avatarMood: "happy",
      retarget: {
        Neck: { z: -0.01, rx: -0.15 }, Neck1: { z: -0.01, rx: -0.15 }, Neck2: { z: -0.01, rx: -0.15 },
        LeftShoulder: { rz: -0.3 }, RightShoulder: { rz: 0.3 }, scaleToEyesLevel: 1.0, origin: { y: -0.1 },
      },
      baseline: { headRotateX: -0.04, eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05, eyesLookDown: 0 },
    },
  },
  ada: {
    name: "Ada",
    voice: "en-maya_f",
    renderer: "splat",
    lipGain: 1,
    jawGain: 0.6,
    asset: "static/avatars/nyx.zip",
    camera: { y: 1.72, z: 0.64, pitchDeg: -6 },
  },
  reachy: {
    name: "Reachy Mini",
    voice: "Aiden",
    renderer: "robot",
    robot: "mini",
    lipGain: 1,
    jawGain: 1,
    asset: "static/avatars/reachy_mini.glb",
    moves: "static/avatars/reachy-moves.json",
  },
  reachy2: {
    name: "Reachy 2",
    voice: "auto-arthur_m",
    renderer: "robot",
    robot: "reachy2",
    lipGain: 1,
    jawGain: 1,
    asset: "static/avatars/reachy2.glb",
    moves: "static/avatars/reachy-moves.json",
  },
  duck: {
    name: "Microduck",
    voice: "auto-clara_f",
    renderer: "robot",
    robot: "duck",
    lipGain: 1,
    jawGain: 1,
    // Its beak follows the mouth shapes, 60 ms ahead of the voice (measured).
    mouth: "beak",
    // It speaks duck: its voice plays as Microduck's own quacks, the words as subtitles.
    quack: true,
    leadMs: 60,
    asset: "static/avatars/robots/duck",
    moves: "static/avatars/reachy-moves.json",
  },
};

/** @param {string} key */
export function coachFor(key) {
  return /** @type {typeof COACHES.max | typeof COACHES.ada | typeof COACHES.reachy} */ (
    COACHES[/** @type {keyof typeof COACHES} */ (key)] || COACHES.max
  );
}
