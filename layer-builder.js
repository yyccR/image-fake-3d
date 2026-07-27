(function exposeDepthLayerBuilder(globalObject) {
  function calculateThresholds(depthValues, layerCount) {
    if (layerCount < 2) return []

    const histogram = new Uint32Array(256)
    for (let index = 0; index < depthValues.length; index += 1) {
      histogram[depthValues[index]] += 1
    }

    const thresholds = []
    let cumulative = 0
    let nextLayer = 1
    for (let value = 0; value < histogram.length && nextLayer < layerCount; value += 1) {
      cumulative += histogram[value]
      while (nextLayer < layerCount && cumulative >= (depthValues.length * nextLayer) / layerCount) {
        thresholds.push(value)
        nextLayer += 1
      }
    }

    while (thresholds.length < layerCount - 1) thresholds.push(255)
    return thresholds
  }

  function assignDepthLayers(depthValues, thresholds) {
    const labels = new Uint8Array(depthValues.length)
    for (let index = 0; index < depthValues.length; index += 1) {
      let layer = 0
      while (layer < thresholds.length && depthValues[index] > thresholds[layer]) layer += 1
      labels[index] = layer
    }
    return labels
  }

  function fillLayerData(sourcePixels, labels, width, height, layerIndex) {
    const pixelCount = width * height
    const output = new Uint8ClampedArray(sourcePixels.length)
    const visited = new Uint8Array(pixelCount)
    const queue = new Int32Array(pixelCount)
    let tail = 0

    for (let pixel = 0; pixel < pixelCount; pixel += 1) {
      if (labels[pixel] !== layerIndex) continue
      const offset = pixel * 4
      output[offset] = sourcePixels[offset]
      output[offset + 1] = sourcePixels[offset + 1]
      output[offset + 2] = sourcePixels[offset + 2]
      output[offset + 3] = 255
      visited[pixel] = 1
      queue[tail] = pixel
      tail += 1
    }

    if (tail === 0) return null

    let head = 0
    const neighbors = new Int32Array(4)
    while (head < tail) {
      const pixel = queue[head]
      head += 1
      const x = pixel % width
      neighbors[0] = x > 0 ? pixel - 1 : -1
      neighbors[1] = x + 1 < width ? pixel + 1 : -1
      neighbors[2] = pixel >= width ? pixel - width : -1
      neighbors[3] = pixel + width < pixelCount ? pixel + width : -1

      for (let index = 0; index < neighbors.length; index += 1) {
        const neighbor = neighbors[index]
        if (neighbor < 0 || visited[neighbor] || labels[neighbor] < layerIndex) continue

        const sourceOffset = pixel * 4
        const targetOffset = neighbor * 4
        output[targetOffset] = output[sourceOffset]
        output[targetOffset + 1] = output[sourceOffset + 1]
        output[targetOffset + 2] = output[sourceOffset + 2]
        output[targetOffset + 3] = 255
        visited[neighbor] = 1
        queue[tail] = neighbor
        tail += 1
      }
    }

    return output
  }

  function createCanvas(width, height) {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    return canvas
  }

  function getImageSize(image) {
    return {
      width: image.naturalWidth || image.videoWidth || image.width,
      height: image.naturalHeight || image.videoHeight || image.height,
    }
  }

  async function build(image, depthImage, options = {}) {
    const layerCount = options.layerCount || 5
    const maxDimension = options.maxDimension || 1280
    const sourceSize = getImageSize(image)
    const scale = Math.min(1, maxDimension / Math.max(sourceSize.width, sourceSize.height))
    const width = Math.max(1, Math.round(sourceSize.width * scale))
    const height = Math.max(1, Math.round(sourceSize.height * scale))

    const sourceCanvas = createCanvas(width, height)
    const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true })
    sourceContext.imageSmoothingEnabled = true
    sourceContext.imageSmoothingQuality = 'high'
    sourceContext.drawImage(image, 0, 0, width, height)
    const sourcePixels = sourceContext.getImageData(0, 0, width, height).data

    const depthCanvas = createCanvas(width, height)
    const depthContext = depthCanvas.getContext('2d', { willReadFrequently: true })
    depthContext.imageSmoothingEnabled = true
    depthContext.imageSmoothingQuality = 'high'
    depthContext.drawImage(depthImage, 0, 0, width, height)
    const depthPixels = depthContext.getImageData(0, 0, width, height).data
    const depthValues = new Uint8Array(width * height)
    for (let pixel = 0; pixel < depthValues.length; pixel += 1) {
      depthValues[pixel] = depthPixels[pixel * 4]
    }

    const thresholds = calculateThresholds(depthValues, layerCount)
    const labels = assignDepthLayers(depthValues, thresholds)
    const layers = []

    for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
      const pixels = fillLayerData(sourcePixels, labels, width, height, layerIndex)
      if (pixels) {
        const canvas = createCanvas(width, height)
        const context = canvas.getContext('2d')
        const imageData = context.createImageData(width, height)
        imageData.data.set(pixels)
        context.putImageData(imageData, 0, 0)
        layers.push({
          canvas,
          depth: (layerIndex + 0.5) / layerCount,
          index: layerIndex,
        })
      }

      options.onProgress?.((layerIndex + 1) / layerCount)
      await new Promise((resolve) => setTimeout(resolve, 0))
    }

    return { height, layers, thresholds, width }
  }

  const api = {
    assignDepthLayers,
    build,
    calculateThresholds,
    fillLayerData,
  }

  globalObject.DepthLayerBuilder = api
  if (typeof module !== 'undefined' && module.exports) module.exports = api
})(typeof globalThis !== 'undefined' ? globalThis : window)
