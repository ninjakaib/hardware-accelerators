/** @type {import('next/config').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/hardware-accelerators",
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
