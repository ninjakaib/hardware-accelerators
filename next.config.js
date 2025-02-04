/** @type {import('next/config').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/hardware-accelerators-site",
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
