import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/vla/:path*',
        destination: 'http://127.0.0.1:6006/api/vla/:path*',
      },
      {
        source: '/api/sae/:path*',
        destination: 'http://127.0.0.1:6006/api/sae/:path*',
      },
      {
        source: '/api/concepts/:path*',
        destination: 'http://127.0.0.1:6006/api/concepts/:path*',
      },
      {
        source: '/api/feature/:path*',
        destination: 'http://127.0.0.1:6006/api/feature/:path*',
      },
      {
        source: '/api/query/:path*',
        destination: 'http://127.0.0.1:6006/api/query/:path*',
      },
      {
        source: '/api/ablation/:path*',
        destination: 'http://127.0.0.1:6006/api/ablation/:path*',
      },
      {
        source: '/api/experiments/:path*',
        destination: 'http://127.0.0.1:6006/api/experiments/:path*',
      },
      {
        source: '/api/layer_features/:path*',
        destination: 'http://127.0.0.1:6006/api/layer_features/:path*',
      },
    ];
  },
};

export default nextConfig;
