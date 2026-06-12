import { userData } from "../data/profile";

const profileStatsBaseUrl = "https://github-profile-summary-cards.vercel.app";
const streakStatsbaseUsr = "https://github-readme-streak-stats.herokuapp.com";
const theme = "bear";

export function getUserMainStats() {
    return `${profileStatsBaseUrl}/api/cards/profile-details?username=${userData.githubUser}&theme=${theme}`;
}

export function getTopLanguageByRepo() {
    return `${profileStatsBaseUrl}/api/cards/repos-per-language?username=${userData.githubUser}&theme=${theme}`;
}

export function getTopLanguageByCommit() {
    return `${profileStatsBaseUrl}/api/cards/most-commit-language?username=${userData.githubUser}&theme=${theme}`;
}

export function getStreaks() {
    return `${streakStatsbaseUsr}/?user=${userData.githubUser}&theme=${theme}&hide_border=true`;
}

export function getProductiveTime() {
    return `${profileStatsBaseUrl}/api/cards/productive-time?username=${userData.githubUser}&theme=${theme}&utcOffset=${userData.timezone}`;
}
