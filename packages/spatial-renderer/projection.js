const FULL_FRAME_DIAGONAL_MM = Math.hypot(36, 24)
const DEFAULT_BASELINE_FRACTION = 0.022
const MAX_BASELINE_FRACTION = 0.07
const REFERENCE_DEPTH_SPREAD = 0.55

export const METRIC_DEPTH_ENCODING = 'metric-depth'
export const RELATIVE_DISPARITY_ENCODING = 'relative-disparity'

export function cameraMetadataFromImage(width, height, focalMm = 30) {
  const safeWidth = positiveNumber(width, 'width')
  const safeHeight = positiveNumber(height, 'height')
  const safeFocalMm = positiveNumber(focalMm, 'focalMm')
  return {
    width: safeWidth,
    height: safeHeight,
    focalPx: safeFocalMm * Math.hypot(safeWidth, safeHeight) / FULL_FRAME_DIAGONAL_MM,
  }
}

export function aspectFitSize(width, height, maxLongEdge) {
  const safeWidth = positiveNumber(width, 'width')
  const safeHeight = positiveNumber(height, 'height')
  const safeLongEdge = positiveNumber(maxLongEdge, 'maxLongEdge')
  const scale = safeLongEdge / Math.max(safeWidth, safeHeight)
  return {
    width: Math.max(1, Math.round(safeWidth * scale)),
    height: Math.max(1, Math.round(safeHeight * scale)),
  }
}

export function mapPoseToSafeBaseline(pose, nominalAmplitude, safeAmplitude) {
  const normalizedPose = finiteNumber(pose, 'pose')
  const nominal = Math.abs(finiteNumber(nominalAmplitude, 'nominalAmplitude'))
  const safe = Math.abs(finiteNumber(safeAmplitude, 'safeAmplitude'))
  return clamp(normalizedPose, -1, 1) * Math.min(nominal, safe)
}

export function dampingAlpha(deltaMs, timeConstantMs = 140) {
  const delta = finiteNumber(deltaMs, 'deltaMs')
  const timeConstant = positiveNumber(timeConstantMs, 'timeConstantMs')
  if (delta <= 0) return 0
  return 1 - Math.exp(-delta / timeConstant)
}

export function depthPreviewDisplayValue(value, encoding = METRIC_DEPTH_ENCODING) {
  const encoded = clamp(Number(value) || 0, 0, 1)
  return encoding === RELATIVE_DISPARITY_ENCODING ? Math.sqrt(encoded) : encoded
}

export function adaptiveBaselineFraction(nearDepth, farDepth) {
  const near = positiveNumber(nearDepth, 'nearDepth')
  const far = positiveNumber(farDepth, 'farDepth')
  const relativeSpread = Math.max(0.04, 1 - Math.min(near, far) / Math.max(near, far))
  return clamp(
    DEFAULT_BASELINE_FRACTION * REFERENCE_DEPTH_SPREAD / relativeSpread,
    DEFAULT_BASELINE_FRACTION,
    MAX_BASELINE_FRACTION,
  )
}

export function limitBaselineForDisocclusion(
  baseline,
  nearDepth,
  farDepth,
  focalPx,
  sourceExtent,
  maxDisocclusionFraction,
) {
  const requested = Number(baseline)
  if (!Number.isFinite(requested)) throw new TypeError('baseline must be finite')
  const near = positiveNumber(nearDepth, 'nearDepth')
  const far = positiveNumber(farDepth, 'farDepth')
  const focal = positiveNumber(focalPx, 'focalPx')
  const extent = positiveNumber(sourceExtent, 'sourceExtent')
  const fraction = Number(maxDisocclusionFraction)
  if (!Number.isFinite(fraction) || fraction <= 0) return requested

  const disparity = Math.abs((1 / near) - (1 / far))
  if (disparity < 1e-6) return requested
  const limit = extent * Math.min(fraction, 0.25) / (focal * disparity)
  return clamp(requested, -limit, limit)
}

export function depthFromPreview(value, nearDepth, farDepth, encoding = METRIC_DEPTH_ENCODING) {
  const near = positiveNumber(nearDepth, 'nearDepth')
  const far = positiveNumber(farDepth, 'farDepth')
  if (far <= near) throw new RangeError('farDepth must be greater than nearDepth')

  const normalized = clamp(Number(value) || 0, 0, 1) ** 2
  if (encoding === RELATIVE_DISPARITY_ENCODING) {
    const disparity = (1 / far) + normalized * ((1 / near) - (1 / far))
    return 1 / disparity
  }
  return far - normalized * (far - near)
}

export function depthMeshGrid(width, height, maxLongEdge = 384) {
  const safeWidth = positiveNumber(width, 'width')
  const safeHeight = positiveNumber(height, 'height')
  const safeMaxLongEdge = Math.max(2, Math.round(positiveNumber(maxLongEdge, 'maxLongEdge')))
  const scale = safeMaxLongEdge / Math.max(safeWidth, safeHeight)
  return {
    columns: Math.max(2, Math.round(safeWidth * scale)),
    rows: Math.max(2, Math.round(safeHeight * scale)),
  }
}

function positiveNumber(value, name) {
  const number = Number(value)
  if (!Number.isFinite(number) || number <= 0) {
    throw new TypeError(`${name} must be a positive finite number`)
  }
  return number
}

function finiteNumber(value, name) {
  const number = Number(value)
  if (!Number.isFinite(number)) throw new TypeError(`${name} must be finite`)
  return number
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}
