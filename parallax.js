(function exposeParallaxMath(globalObject) {
  const layers = Object.freeze([
    Object.freeze({ id: 'sky', label: '天空', depth: '∞', maxShift: 1.5 }),
    Object.freeze({ id: 'far', label: '远山', depth: 12, maxShift: 5 }),
    Object.freeze({ id: 'middle', label: '中景', depth: 6, maxShift: 11 }),
    Object.freeze({ id: 'near', label: '前景', depth: 2, maxShift: 24 }),
  ])

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value))
  }

  function normalizePointer(value, start, size) {
    if (size <= 0) return 0
    return clamp(((value - start) / size) * 2 - 1, -1, 1)
  }

  function calculateOffset(x, y, maxShift, intensity) {
    const safeX = clamp(x, -1, 1)
    const safeY = clamp(y, -1, 1)
    const safeIntensity = clamp(intensity == null ? 1 : intensity, 0, 1.5)

    return {
      x: safeX * maxShift * safeIntensity,
      y: safeY * maxShift * 0.72 * safeIntensity,
    }
  }

  function createParallaxSnapshot(x, y, intensity) {
    return Object.fromEntries(
      layers.map((layer) => [
        layer.id,
        calculateOffset(x, y, layer.maxShift, intensity),
      ]),
    )
  }

  const api = {
    PARALLAX_LAYERS: layers,
    calculateOffset,
    clamp,
    createParallaxSnapshot,
    normalizePointer,
  }

  globalObject.ParallaxMath = api

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api
  }
})(typeof globalThis !== 'undefined' ? globalThis : window)
