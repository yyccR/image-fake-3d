const test = require('node:test')
const assert = require('node:assert/strict')

const {
  assignDepthLayers,
  calculateThresholds,
  fillLayerData,
} = require('../layer-builder.js')

test('splits a depth field into ordered quantile layers', () => {
  const depth = Uint8Array.from([0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
  const thresholds = calculateThresholds(depth, 5)
  const labels = assignDepthLayers(depth, thresholds)

  assert.equal(thresholds.length, 4)
  assert.deepEqual(Array.from(labels), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
})

test('fills occluded pixels from the same rear plane without covering farther pixels', () => {
  const colors = Uint8ClampedArray.from([
    10, 20, 30, 255,
    90, 100, 110, 255,
    180, 190, 200, 255,
  ])
  const labels = Uint8Array.from([0, 1, 2])

  const background = fillLayerData(colors, labels, 3, 1, 0)
  const middle = fillLayerData(colors, labels, 3, 1, 1)

  assert.deepEqual(Array.from(background), [
    10, 20, 30, 255,
    10, 20, 30, 255,
    10, 20, 30, 255,
  ])
  assert.deepEqual(Array.from(middle), [
    0, 0, 0, 0,
    90, 100, 110, 255,
    90, 100, 110, 255,
  ])
})
