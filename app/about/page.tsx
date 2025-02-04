import AboutUs from "./About";
import Header from "../components/Header";
import Footer from "../components/Footer";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main>
        <AboutUs />
      </main>
      <Footer />
    </div>
  );
}
