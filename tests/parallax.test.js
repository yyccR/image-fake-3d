const test = require('node:test')
const assert = require('node:assert/strict')
const {
  calculateOffset,
  createParallaxSnapshot,
  normalizePointer,
} = require('../parallax.js')

test('normalizes pointer coordinates around the viewport center', () => {
  assert.equal(normalizePointer(50, 0, 100), 0)
  assert.equal(normalizePointer(0, 0, 100), -1)
  assert.equal(normalizePointer(100, 0, 100), 1)
})

test('clamps pointer input and scales vertical movement', () => {
  const offset = calculateOffset(2, -2, 20, 1)
  assert.equal(offset.x, 20)
  assert.ok(Math.abs(offset.y + 14.4) < Number.EPSILON * 10)
})

test('moves near layers farther than distant layers', () => {
  const snapshot = createParallaxSnapshot(0.8, 0, 1)
  assert.ok(Math.abs(snapshot.near.x) > Math.abs(snapshot.middle.x))
  assert.ok(Math.abs(snapshot.middle.x) > Math.abs(snapshot.far.x))
  assert.ok(Math.abs(snapshot.far.x) > Math.abs(snapshot.sky.x))
})

test('scales all displacement with the selected intensity', () => {
  const normal = createParallaxSnapshot(0.5, 0.25, 1)
  const half = createParallaxSnapshot(0.5, 0.25, 0.5)
  assert.equal(half.near.x, normal.near.x / 2)
  assert.equal(half.middle.y, normal.middle.y / 2)
})
