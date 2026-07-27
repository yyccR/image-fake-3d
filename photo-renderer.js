(function exposeDepthPhotoRenderer(globalObject) {
  const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_uv;

    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
      v_uv = a_position * 0.5 + 0.5;
    }
  `

  const fragmentShaderSource = `
    precision mediump float;

    uniform sampler2D u_image;
    uniform sampler2D u_depth;
    uniform vec2 u_pose;
    uniform vec2 u_uv_scale;
    uniform bool u_has_depth;
    uniform bool u_show_depth;
    varying vec2 v_uv;

    void main() {
      vec2 base_uv = (v_uv - 0.5) * (u_uv_scale / 1.06) + 0.5;
      float procedural_depth = smoothstep(0.02, 0.98, 1.0 - v_uv.y);
      float supplied_depth = texture2D(u_depth, base_uv).r;
      float depth = u_has_depth ? supplied_depth : procedural_depth;

      if (u_show_depth) {
        gl_FragColor = vec4(vec3(depth), 1.0);
        return;
      }

      vec2 shifted_uv = base_uv + u_pose * mix(0.12, 1.0, depth);
      shifted_uv = clamp(shifted_uv, vec2(0.004), vec2(0.996));
      gl_FragColor = texture2D(u_image, shifted_uv);
    }
  `

  const layerFragmentShaderSource = `
    precision mediump float;

    uniform sampler2D u_layer;
    uniform vec2 u_pose;
    uniform vec2 u_uv_scale;
    varying vec2 v_uv;

    void main() {
      vec2 base_uv = (v_uv - 0.5) * (u_uv_scale / 1.06) + 0.5;
      vec2 shifted_uv = clamp(base_uv + u_pose, vec2(0.004), vec2(0.996));
      gl_FragColor = texture2D(u_layer, shifted_uv);
    }
  `

  function createShader(gl, type, source) {
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader)
      gl.deleteShader(shader)
      throw new Error(`Shader compilation failed: ${message}`)
    }
    return shader
  }

  function createProgram(gl, fragmentSource) {
    const program = gl.createProgram()
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource)
    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)
    gl.deleteShader(vertexShader)
    gl.deleteShader(fragmentShader)

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const message = gl.getProgramInfoLog(program)
      gl.deleteProgram(program)
      throw new Error(`Shader linking failed: ${message}`)
    }
    return program
  }

  function createTexture(gl, pixel) {
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array(pixel),
    )
    return texture
  }

  class DepthPhotoRenderer {
    constructor(canvas) {
      this.canvas = canvas
      this.gl = canvas.getContext('webgl', {
        alpha: false,
        antialias: false,
        depth: false,
        preserveDrawingBuffer: false,
      })
      this.imageAspect = 1
      this.hasImage = false
      this.hasDepth = false
      this.imageSource = null
      this.layerTextures = []
      this.layerBuildId = 0

      if (!this.gl) throw new Error('This browser does not support WebGL.')

      const gl = this.gl
      this.program = createProgram(gl, fragmentShaderSource)
      this.positionLocation = gl.getAttribLocation(this.program, 'a_position')
      this.poseLocation = gl.getUniformLocation(this.program, 'u_pose')
      this.uvScaleLocation = gl.getUniformLocation(this.program, 'u_uv_scale')
      this.hasDepthLocation = gl.getUniformLocation(this.program, 'u_has_depth')
      this.showDepthLocation = gl.getUniformLocation(this.program, 'u_show_depth')
      this.imageSamplerLocation = gl.getUniformLocation(this.program, 'u_image')
      this.depthSamplerLocation = gl.getUniformLocation(this.program, 'u_depth')

      this.layerProgram = createProgram(gl, layerFragmentShaderSource)
      this.layerPositionLocation = gl.getAttribLocation(this.layerProgram, 'a_position')
      this.layerPoseLocation = gl.getUniformLocation(this.layerProgram, 'u_pose')
      this.layerUvScaleLocation = gl.getUniformLocation(this.layerProgram, 'u_uv_scale')
      this.layerSamplerLocation = gl.getUniformLocation(this.layerProgram, 'u_layer')

      this.positionBuffer = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer)
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
        gl.STATIC_DRAW,
      )

      this.imageTexture = createTexture(gl, [31, 78, 79, 255])
      this.depthTexture = createTexture(gl, [127, 127, 127, 255])
    }

    uploadTexture(texture, image) {
      const gl = this.gl
      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image)
    }

    setImage(image) {
      const width = image.naturalWidth || image.width
      const height = image.naturalHeight || image.height
      this.imageAspect = width / height
      this.imageSource = image
      this.layerBuildId += 1
      this.clearLayerTextures()
      this.uploadTexture(this.imageTexture, image)
      this.hasImage = true
    }

    async setDepthImage(image, onProgress) {
      this.uploadTexture(this.depthTexture, image)
      this.hasDepth = true
      const buildId = ++this.layerBuildId
      this.clearLayerTextures()

      if (!this.imageSource || !globalObject.DepthLayerBuilder) return 0

      let result
      try {
        result = await globalObject.DepthLayerBuilder.build(this.imageSource, image, {
          layerCount: 5,
          maxDimension: 1280,
          onProgress,
        })
      } catch (error) {
        console.warn('Layered scene construction failed; using continuous depth rendering:', error)
        return 0
      }
      if (buildId !== this.layerBuildId) return 0

      this.layerTextures = result.layers.map((layer) => {
        const texture = createTexture(this.gl, [0, 0, 0, 0])
        this.uploadTexture(texture, layer.canvas)
        return { depth: layer.depth, texture }
      })
      return this.layerTextures.length
    }

    clearDepthImage() {
      this.layerBuildId += 1
      this.hasDepth = false
      this.clearLayerTextures()
    }

    clearLayerTextures() {
      this.layerTextures.forEach((layer) => this.gl.deleteTexture(layer.texture))
      this.layerTextures = []
    }

    resize() {
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2)
      const width = Math.max(1, Math.round(this.canvas.clientWidth * pixelRatio))
      const height = Math.max(1, Math.round(this.canvas.clientHeight * pixelRatio))
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width
        this.canvas.height = height
      }
    }

    getUvScale() {
      const canvasAspect = this.canvas.width / this.canvas.height
      let scaleX = 1
      let scaleY = 1
      if (this.imageAspect > canvasAspect) {
        scaleX = canvasAspect / this.imageAspect
      } else {
        scaleY = this.imageAspect / canvasAspect
      }
      return { scaleX, scaleY }
    }

    renderContinuous(x, y, intensity, showDepth, scaleX, scaleY) {
      const gl = this.gl

      gl.disable(gl.BLEND)
      gl.useProgram(this.program)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer)
      gl.enableVertexAttribArray(this.positionLocation)
      gl.vertexAttribPointer(this.positionLocation, 2, gl.FLOAT, false, 0, 0)

      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, this.imageTexture)
      gl.uniform1i(this.imageSamplerLocation, 0)
      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, this.depthTexture)
      gl.uniform1i(this.depthSamplerLocation, 1)

      gl.uniform2f(this.poseLocation, x * intensity * 0.022, y * intensity * 0.015)
      gl.uniform2f(this.uvScaleLocation, scaleX, scaleY)
      gl.uniform1i(this.hasDepthLocation, this.hasDepth ? 1 : 0)
      gl.uniform1i(this.showDepthLocation, showDepth ? 1 : 0)
      gl.drawArrays(gl.TRIANGLES, 0, 6)
    }

    renderLayers(x, y, intensity, scaleX, scaleY) {
      const gl = this.gl
      gl.clearColor(0.03, 0.09, 0.09, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.enable(gl.BLEND)
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
      gl.useProgram(this.layerProgram)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer)
      gl.enableVertexAttribArray(this.layerPositionLocation)
      gl.vertexAttribPointer(this.layerPositionLocation, 2, gl.FLOAT, false, 0, 0)
      gl.uniform2f(this.layerUvScaleLocation, scaleX, scaleY)
      gl.uniform1i(this.layerSamplerLocation, 0)

      this.layerTextures.forEach((layer) => {
        const shift = 0.08 + layer.depth * 0.92
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, layer.texture)
        gl.uniform2f(
          this.layerPoseLocation,
          x * intensity * 0.022 * shift,
          y * intensity * 0.015 * shift,
        )
        gl.drawArrays(gl.TRIANGLES, 0, 6)
      })

      gl.disable(gl.BLEND)
    }

    render(x, y, intensity, showDepth) {
      if (!this.hasImage) return
      this.resize()

      const gl = this.gl
      const { scaleX, scaleY } = this.getUvScale()
      gl.viewport(0, 0, this.canvas.width, this.canvas.height)

      if (!showDepth && this.layerTextures.length > 1) {
        this.renderLayers(x, y, intensity, scaleX, scaleY)
      } else {
        this.renderContinuous(x, y, intensity, showDepth, scaleX, scaleY)
      }
    }
  }

  globalObject.DepthPhotoRenderer = DepthPhotoRenderer
})(typeof globalThis !== 'undefined' ? globalThis : window)
