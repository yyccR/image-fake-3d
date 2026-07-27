const test = require('node:test')
const assert = require('node:assert/strict')

test('derives the same field of view for proportionally resized images', async () => {
  const { cameraMetadataFromImage } = await import('../packages/spatial-renderer/projection.js')
  const small = cameraMetadataFromImage(1080, 1920)
  const large = cameraMetadataFromImage(2160, 3840)

  assert.equal(large.width, small.width * 2)
  assert.equal(large.height, small.height * 2)
  assert.ok(Math.abs(large.focalPx - small.focalPx * 2) < Number.EPSILON * 1000)
})

test('rejects invalid image dimensions', async () => {
  const { cameraMetadataFromImage } = await import('../packages/spatial-renderer/projection.js')
  assert.throws(() => cameraMetadataFromImage(0, 1080), /width/)
  assert.throws(() => cameraMetadataFromImage(1920, Number.NaN), /height/)
})
