import Link from "next/link";

export default function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-black bg-opacity-50 backdrop-blur-md">
      <nav className="container mx-auto px-4 py-4">
        <ul className="flex justify-center space-x-8">
          <li>
            <Link href="/" className="text-sm font-medium hover:text-gray-300">
              Home
            </Link>
          </li>
          <li>
            <Link href="#" className="text-sm font-medium hover:text-gray-300">
              Products
            </Link>
          </li>
          <li>
            <Link
              href="/about"
              className="text-sm font-medium hover:text-gray-300"
            >
              About
            </Link>
          </li>
        </ul>
      </nav>
    </header>
  );
}
