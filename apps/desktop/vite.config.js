import { defineConfig } from 'vite'

export default defineConfig({
  root: 'src',
  server: {
    port: 1420,
    strictPort: true,
  },
  clearScreen: false,
  build: {
    target: 'safari15',
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: 'src/index.html',
        wallpaper: 'src/wallpaper.html',
      },
    },
  },
})
