import Image from "next/image";
import { Github, Linkedin, Twitter } from "lucide-react";

interface Socials {
  twitter?: string;
  linkedin?: string;
  github?: string;
}

interface Person {
  name: string;
  role: string;
  image: string;
  bio: string;
  socials: Socials;
}

const people: Person[] = [
  {
    name: "Dr. Rajesh Gupta",
    role: "Mentor",
    image: "/hardware-accelerators-site/rajesh.jpg?height=400&width=400",
    bio: "Founding Director at HDSI. Distinguished Professor at UCSD.",
    socials: {
      linkedin: "https://linkedin.com/in/rajeshgupta4",
    },
  },
  {
    name: "Kai Breese",
    role: "De-facto Captain",
    image: "/hardware-accelerators-site/kai.jpg?height=400&width=400",
    bio: "[bio]",
    socials: {
      linkedin: "https://linkedin.com/in/kaibreese",
      github: "https://github.com/ninjakaib",
    },
  },
  {
    name: "Justin Chou",
    role: "Code Monkey",
    image: "/hardware-accelerators-site/justin.jpg?height=400&width=400",
    bio: "[bio]",
    socials: {
      linkedin: "https://www.linkedin.com/in/justintchou/",
      github: "https://github.com/nakschou",
    },
  },
  {
    name: "Katelyn Abille",
    role: "Figma God",
    image: "/hardware-accelerators-site/katelyn.jpg?height=400&width=400",
    bio: "[bio]",
    socials: {
      linkedin: "https://linkedin.com/in/katelynmaea",
      github: "https://github.com/katemae",
    },
  },
  {
    name: "Lukas Fullner",
    role: "Verilog Enjoyer (somehow)",
    image: "/hardware-accelerators-site/lukas.jpg?height=400&width=400",
    bio: "[bio]",
    socials: {
      linkedin: "https://linkedin.com/in/lukas-fullner-172639284",
      github: "https://github.com/Lwizard3",
    },
  },
];

export default function AboutUs() {
  return (
    <>
      <section className="relative flex items-center justify-center py-16 bg-zinc-900">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold text-center mb-8">
            About Our Team
          </h1>

          {/* Mentor Section */}
          <div className="mb-12">
            <h2 className="text-2xl font-semibold mb-4">Our Mentor</h2>
            <PersonCard {...people[0]} />
          </div>

          {/* Team Members Section */}
          <div>
            <h2 className="text-2xl font-semibold mb-4">Our Team</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {people.slice(1).map((person, index) => (
                <PersonCard key={index} {...person} />
              ))}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

function PersonCard({ name, role, image, bio, socials }: Person) {
  return (
    <div className="bg-white shadow-lg rounded-lg overflow-hidden">
      <div className="md:flex">
        <div className="md:flex-shrink-0">
          <Image
            className="h-48 w-full object-cover md:w-48"
            src={image || "/placeholder.svg"}
            alt={name}
            width={200}
            height={200}
          />
        </div>
        <div className="p-8">
          <div className="uppercase tracking-wide text-sm text-indigo-500 font-semibold">
            {role}
          </div>
          <h3 className="mt-1 text-lg font-medium text-gray-900">{name}</h3>
          <p className="mt-2 text-gray-600">{bio}</p>
          <div className="mt-4 flex space-x-4">
            {socials.twitter && (
              <a
                href={socials.twitter}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-500"
              >
                <Twitter size={20} />
              </a>
            )}
            {socials.linkedin && (
              <a
                href={socials.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-700"
              >
                <Linkedin size={20} />
              </a>
            )}
            {socials.github && (
              <a
                href={socials.github}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-gray-900"
              >
                <Github size={20} />
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
