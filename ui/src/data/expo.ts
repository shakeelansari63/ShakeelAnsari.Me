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
    name: "NSE-Chatbot",
    description: "Chatbot for NSE related info.",
    appUrl: "https://nse-chatbot.azurewebsites.net/",
    codeUrl: "https://github.com/shakeelansari63/NSE-Chatbot",
    productPageUrl: "/product/nse-chatbot",
    thumbnail: "",
  },
  {
    name: "Poke-Budgy",
    description: "Budget Planner in your pocket.",
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
