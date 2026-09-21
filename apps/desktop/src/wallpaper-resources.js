const WALLPAPER_IMAGE_RESOURCES = [
  { urlKey: 'depthUrl', imageKey: 'depthImage', label: '深度图' },
  { urlKey: 'backgroundUrl', imageKey: 'backgroundImage', label: '补洞背景' },
  { urlKey: 'subjectUrl', imageKey: 'subjectImage', label: '主体边界' },
]

function resourceError(label, status) {
  const detail = status ? `（HTTP ${status}）` : ''
  return new Error(`无法读取本机生成的${label}${detail}。`)
}

async function fetchResourceBlob(url, label, fetchImpl) {
  let response
  try {
    response = await fetchImpl(url, { cache: 'no-store' })
  } catch {
    throw resourceError(label)
  }
  if (!response.ok) throw resourceError(label, response.status)
  try {
    return await response.blob()
  } catch {
    throw resourceError(label)
  }
}

function loadResourceImage(objectUrl, label) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.decoding = 'async'
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error(`无法解码本机生成的${label}。`))
    image.src = objectUrl
  })
}

export async function prepareWallpaperImageResources(
  metadata,
  sceneSource,
  {
    fetchImpl = globalThis.fetch,
    urlApi = globalThis.URL,
    imageLoader = loadResourceImage,
  } = {},
) {
  const preparedMetadata = { ...metadata }
  const objectUrls = []
  let released = false

  const release = () => {
    if (released) return
    released = true
    for (const objectUrl of objectUrls) {
      try {
        urlApi.revokeObjectURL(objectUrl)
      } catch {
        // Cleanup must not hide the original loading or rendering error.
      }
    }
  }

  try {
    for (const resource of WALLPAPER_IMAGE_RESOURCES) {
      const path = metadata[resource.urlKey]
      if (!path) continue
      const sourceUrl = new URL(path, sceneSource).href
      const blob = await fetchResourceBlob(sourceUrl, resource.label, fetchImpl)
      const objectUrl = urlApi.createObjectURL(blob)
      objectUrls.push(objectUrl)
      preparedMetadata[resource.imageKey] = await imageLoader(objectUrl, resource.label)
    }
  } catch (error) {
    release()
    throw error
  }

  return { metadata: preparedMetadata, release }
}
