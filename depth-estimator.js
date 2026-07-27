(function exposeBrowserDepthEstimator(globalObject) {
  const LIBRARY_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm'
  const MODEL_ID = 'onnx-community/depth-anything-v2-small'

  let estimatorPromise = null
  let progressListener = null

  async function loadEstimator(onProgress) {
    progressListener = onProgress

    if (!estimatorPromise) {
      estimatorPromise = import(LIBRARY_URL)
        .then(({ env, pipeline }) => {
          env.allowLocalModels = false
          return pipeline('depth-estimation', MODEL_ID, {
            device: 'wasm',
            dtype: 'q8',
            progress_callback: (event) => progressListener?.(event),
          })
        })
        .catch((error) => {
          estimatorPromise = null
          throw error
        })
    }

    return estimatorPromise
  }

  async function estimate(sourceUrl, onProgress) {
    if (!sourceUrl) throw new Error('A source image URL is required.')

    const estimator = await loadEstimator(onProgress)
    progressListener = onProgress
    progressListener?.({ status: 'inference' })
    const result = await estimator(sourceUrl)
    if (!result?.depth?.toCanvas) throw new Error('The model did not return a depth image.')
    return result.depth.toCanvas()
  }

  globalObject.BrowserDepthEstimator = {
    estimate,
    modelId: MODEL_ID,
  }
})(typeof globalThis !== 'undefined' ? globalThis : window)
