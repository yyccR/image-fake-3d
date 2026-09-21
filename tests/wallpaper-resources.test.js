const test = require('node:test')
const assert = require('node:assert/strict')

const resourcesModule = import('../apps/desktop/src/wallpaper-resources.js')

async function prepareWallpaperImageResources(...args) {
  const resources = await resourcesModule
  return resources.prepareWallpaperImageResources(...args)
}

function createUrlApi() {
  const created = []
  const revoked = []
  return {
    created,
    revoked,
    createObjectURL(blob) {
      created.push(blob)
      return `blob:wallpaper-${created.length}`
    },
    revokeObjectURL(url) {
      revoked.push(url)
    },
  }
}

test('converts generated wallpaper images to CSP-safe blob URLs', async () => {
  const requests = []
  const urlApi = createUrlApi()
  const metadata = {
    depthUrl: './depth',
    backgroundUrl: '/api/jobs/job-1/background',
    subjectUrl: 'http://127.0.0.1:4173/api/jobs/job-1/subject',
    width: 1920,
  }
  const blobs = [new Blob(['depth']), new Blob(['background']), new Blob(['subject'])]
  const images = [{ id: 'depth' }, { id: 'background' }, { id: 'subject' }]
  const loadedUrls = []
  const fetchImpl = async (url, options) => {
    requests.push({ url, options })
    return { ok: true, blob: async () => blobs[requests.length - 1] }
  }
  const imageLoader = async (url) => {
    loadedUrls.push(url)
    return images[loadedUrls.length - 1]
  }

  const prepared = await prepareWallpaperImageResources(
    metadata,
    'http://127.0.0.1:4173/api/jobs/job-1/result',
    { fetchImpl, urlApi, imageLoader },
  )

  assert.deepEqual(requests, [
    {
      url: 'http://127.0.0.1:4173/api/jobs/job-1/depth',
      options: { cache: 'no-store' },
    },
    {
      url: 'http://127.0.0.1:4173/api/jobs/job-1/background',
      options: { cache: 'no-store' },
    },
    {
      url: 'http://127.0.0.1:4173/api/jobs/job-1/subject',
      options: { cache: 'no-store' },
    },
  ])
  assert.deepEqual(prepared.metadata, {
    ...metadata,
    depthImage: images[0],
    backgroundImage: images[1],
    subjectImage: images[2],
  })
  assert.deepEqual(urlApi.created, blobs)
  assert.deepEqual(loadedUrls, [
    'blob:wallpaper-1',
    'blob:wallpaper-2',
    'blob:wallpaper-3',
  ])
  assert.notEqual(prepared.metadata, metadata)

  prepared.release()
  prepared.release()
  assert.deepEqual(urlApi.revoked, [
    'blob:wallpaper-1',
    'blob:wallpaper-2',
    'blob:wallpaper-3',
  ])
})

test('skips optional wallpaper image resources that are absent', async () => {
  const urlApi = createUrlApi()
  let fetchCount = 0
  const metadata = { width: 1280, height: 720 }
  const prepared = await prepareWallpaperImageResources(metadata, 'file:///scene.sog', {
    fetchImpl: async () => {
      fetchCount += 1
      throw new Error('unexpected request')
    },
    urlApi,
    imageLoader: async () => {
      throw new Error('unexpected image decode')
    },
  })

  assert.deepEqual(prepared.metadata, metadata)
  assert.equal(fetchCount, 0)
  prepared.release()
  assert.deepEqual(urlApi.revoked, [])
})

test('names the failed resource and cleans earlier blob URLs after an HTTP error', async () => {
  const urlApi = createUrlApi()
  let fetchCount = 0
  const fetchImpl = async () => {
    fetchCount += 1
    if (fetchCount === 1) return { ok: true, blob: async () => new Blob(['depth']) }
    return { ok: false, status: 503 }
  }

  await assert.rejects(
    prepareWallpaperImageResources(
      {
        depthUrl: '/depth',
        backgroundUrl: '/background',
        subjectUrl: '/subject',
      },
      'http://127.0.0.1:4173/api/jobs/job-1/result',
      { fetchImpl, urlApi, imageLoader: async (url) => ({ src: url }) },
    ),
    { message: '无法读取本机生成的补洞背景（HTTP 503）。' },
  )
  assert.equal(fetchCount, 2)
  assert.deepEqual(urlApi.revoked, ['blob:wallpaper-1'])
})

test('reports a subject boundary network failure without leaking object URLs', async () => {
  const urlApi = createUrlApi()
  let fetchCount = 0
  const fetchImpl = async () => {
    fetchCount += 1
    if (fetchCount < 3) return { ok: true, blob: async () => new Blob(['ok']) }
    throw new TypeError('connection closed')
  }

  await assert.rejects(
    prepareWallpaperImageResources(
      {
        depthUrl: '/depth',
        backgroundUrl: '/background',
        subjectUrl: '/subject',
      },
      'http://127.0.0.1:4173/api/jobs/job-1/result',
      { fetchImpl, urlApi, imageLoader: async (url) => ({ src: url }) },
    ),
    { message: '无法读取本机生成的主体边界。' },
  )
  assert.deepEqual(urlApi.revoked, ['blob:wallpaper-1', 'blob:wallpaper-2'])
})

test('names a decode failure and revokes the broken image URL', async () => {
  const urlApi = createUrlApi()

  await assert.rejects(
    prepareWallpaperImageResources(
      { depthUrl: '/depth' },
      'http://127.0.0.1:4173/api/jobs/job-1/result',
      {
        fetchImpl: async () => ({ ok: true, blob: async () => new Blob(['depth']) }),
        urlApi,
        imageLoader: async () => {
          throw new Error('无法解码本机生成的深度图。')
        },
      },
    ),
    { message: '无法解码本机生成的深度图。' },
  )
  assert.deepEqual(urlApi.revoked, ['blob:wallpaper-1'])
})
