export interface ExpoProject {
  name: string;
  description: string;
  appUrl?: string;
  codeUrl?: string;
  productPageUrl?: string;
  thumbnail: string;
}

export const expo: ExpoProject[] = [
  {
    name: "Guitarists",
    description:
      "Open-source, offline-first desktop toolkit with tuner, metronome, tab editor, and songwriting notebook.",
    appUrl: "https://github.com/shakeelansari63/Guitarists/releases",
    codeUrl: "https://github.com/shakeelansari63/Guitarists",
    productPageUrl: "/product/guitarists",
    thumbnail: "",
  },
  {
    name: "NSE-Chatbot",
    description:
      "AI-powered market assistant for NSE India with live quotes, trend analysis, sector searches, and company financial summaries.",
    appUrl: "https://nse-chatbot.azurewebsites.net/",
    codeUrl: "https://github.com/shakeelansari63/NSE-Chatbot",
    productPageUrl: "/product/nse-chatbot",
    thumbnail: "",
  },
  {
    name: "Poke-Budgy",
    description:
      "Minimalist, 100% offline Android budget manager with no sign-up or tracking, quick cash-flow tracking, and clear visual insights.",
    appUrl:
      "https://play.google.com/store/apps/details?id=com.shakeelansari63.pokebudgy",
    codeUrl: "https://github.com/shakeelansari63/poke-budgy",
    productPageUrl: "/product/poke-budgy",
    thumbnail: "",
  },
  {
    name: "MyTrA",
    description:
      "My True Assistant. AI assistant for your mobile with power of MCP and Agents.",
    codeUrl: "https://github.com/shakeelansari63/MyTrA",
    thumbnail: "",
  },
  {
    name: "Rocket-Invo",
    description: "Billing software for small scale business.",
    codeUrl: "https://github.com/shakeelansari63/Rocket-Invo",
    thumbnail: "",
  },
  {
    name: "SQLyser",
    description: "Analyse your SQL Queries.",
    codeUrl: "https://github.com/shakeelansari63/sqlyser",
    thumbnail: "",
  },
  {
    name: "ShakeelAnsari.Me",
    description:
      "Personal Portfolio App. Website written in PHP and React for Personal Portfolio showcase.",
    codeUrl: "https://github.com/shakeelansari63/ShakeelAnsari.Me",
    thumbnail: "",
  },
];
