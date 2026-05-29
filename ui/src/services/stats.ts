import { userData } from './data';

const profileStatsBaseUrl = 'https://github-profile-summary-cards.vercel.app';
const githubStatsbaseUrl = 'https://github-readme-stats.vercel.app';
const streakStatsbaseUsr = 'https://github-readme-streak-stats.herokuapp.com';
const starDevbaseUrl = 'https://stardev.io';
const theme = 'bear';

export function getUserMainStats() {
  return `${profileStatsBaseUrl}/api/cards/profile-details?username=${userData.githubUser}&theme=${theme}`;
}

export function getUserGithubStats() {
  return `${githubStatsbaseUrl}/api?username=${userData.githubUser}&show_icons=true&include_all_commits=true&theme=${theme}&hide_border=true`;
}

export function getUserGithubStars() {
  return `${githubStatsbaseUrl}/api?username=${userData.githubUser}&show_icons=true&include_all_commits=true&theme=${theme}&hide_border=true&show=reviews,discussions_started,discussions_answered,prs_merged,prs_merged_percentage&hide=stars,commits,prs,issues,contribs`;
}

export function getTopLanguageByRepo() {
  return `${profileStatsBaseUrl}/api/cards/repos-per-language?username=${userData.githubUser}&theme=${theme}`;
}

export function getTopLanguageByCommit() {
  return `${profileStatsBaseUrl}/api/cards/most-commit-language?username=${userData.githubUser}&theme=${theme}`;
}

export function getTopLanguages() {
  return `${githubStatsbaseUrl}/api/top-langs/?username=${userData.githubUser}&layout=compact&theme=${theme}&hide_border=true&&langs_count=8`;
}

export function getStreaks() {
  return `${streakStatsbaseUsr}/?user=${userData.githubUser}&theme=${theme}&hide_border=true`;
}

export function getProductiveTime() {
  return `${profileStatsBaseUrl}/api/cards/productive-time?username=${userData.githubUser}&theme=${theme}&utcOffset=${userData.timezone}`;
}

export function getLocalStarDevStats() {
  return `${starDevbaseUrl}/developers/${userData.githubUser}/badge/languages/locality.svg`;
}

export function getGlobalStarDevStats() {
  return `${starDevbaseUrl}/developers/${userData.githubUser}/badge/languages/global.svg`;
}

export function getProjectCard(repo: string) {
  return `${githubStatsbaseUrl}/api/pin/?username=${userData.githubUser}&repo=${repo}&theme=${theme}&hide_border=true`;
}
