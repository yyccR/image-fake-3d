const FULL_FRAME_DIAGONAL_MM = Math.hypot(36, 24)

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

function positiveNumber(value, name) {
  const number = Number(value)
  if (!Number.isFinite(number) || number <= 0) {
    throw new TypeError(`${name} must be a positive finite number`)
  }
  return number
}
