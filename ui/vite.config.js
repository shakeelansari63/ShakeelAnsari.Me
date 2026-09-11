import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  css: {
    preprocessorOptions: {
      scss: {
        api: 'modern-compiler',
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          // NOTE: antd and @ant-design/icons must stay in ONE chunk — antd
          // components import icons internally, so splitting them creates a
          // vendor-antd <-> vendor-antd-icons cycle that crashes at load
          // ("Cannot read properties of undefined (reading 'primary')").
          if (id.includes('node_modules/antd') || id.includes('node_modules/rc-') || id.includes('node_modules/@ant-design')) {
            return 'vendor-antd';
          }
          if (id.includes('node_modules/react-syntax-highlighter')) return 'vendor-highlight';
          if (id.includes('node_modules/recharts') || id.includes('node_modules/d3-')) return 'vendor-charts';
          if (
            id.includes('node_modules/react/') ||
            id.includes('node_modules/react-dom') ||
            id.includes('node_modules/react-router') ||
            id.includes('node_modules/react-helmet-async') ||
            id.includes('node_modules/scheduler')
          )
            return 'vendor-react';
        },
      },
    },
  },
  server: {
    port: 3000,
    historyApiFallback: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
