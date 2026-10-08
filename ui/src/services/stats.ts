import { userData } from "../data/profile";
import { languageColor } from "../data/languageColors";
import { fetchContributionCounts } from "./api";
import type {
  ContributionData,
  GitProfile,
  LanguageStat,
  ProfileStats,
  StreakRange,
  StreakStats,
} from "../models/types";

const githubApiUrl = "https://api.github.com";
const contributionApiUrl = "/api/github/contributions";

const MAX_REPO_PAGES = 3;
const MAX_COMMIT_PAGES = 5;
const MAX_ORGS = 3;
const TOP_LANGUAGES = 5;
const DAY_MS = 24 * 60 * 60 * 1000;
const MIN_CONTRIBUTION_YEAR = 2008;

interface GitHubRepo {
  full_name: string;
  language: string | null;
  fork: boolean;
}

interface OrgRepo {
  full_name: string;
  language: string | null;
}

interface CommitSearchItem {
  repository: { full_name: string };
  commit: { author: { date: string } };
}

function cached<T>(fetcher: () => Promise<T | null>): () => Promise<T | null> {
  let promise: Promise<T | null> | null = null;
  return () => {
    if (!promise) {
      promise = fetcher().then((value) => {
        if (value === null) promise = null;
        return value;
      });
    }
    return promise;
  };
}

const getProfile = cached(async (): Promise<GitProfile | null> => {
  try {
    const res = await fetch(`${githubApiUrl}/users/${userData.githubUser}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
});

const getOwnedRepos = cached(async (): Promise<GitHubRepo[] | null> => {
  const repos: GitHubRepo[] = [];
  try {
    for (let page = 1; page <= MAX_REPO_PAGES; page++) {
      const res = await fetch(
        `${githubApiUrl}/users/${userData.githubUser}/repos?type=owner&per_page=100&page=${page}`,
      );
      if (!res.ok) break;
      const batch: GitHubRepo[] = await res.json();
      repos.push(...batch);
      if (batch.length < 100) break;
    }
  } catch {
    return repos.length > 0 ? repos : null;
  }
  return repos.length > 0 ? repos : null;
});

const getOrgLogins = cached(async (): Promise<string[] | null> => {
  try {
    const res = await fetch(`${githubApiUrl}/users/${userData.githubUser}/orgs`);
    if (!res.ok) return null;
    const orgs: { login: string }[] = await res.json();
    return orgs.slice(0, MAX_ORGS).map((org) => org.login);
  } catch {
    return null;
  }
});

const getOrgRepoLanguages = cached(
  async (): Promise<Map<string, string> | null> => {
    try {
      const orgs = await getOrgLogins();
      if (orgs === null) return null;
      const languages = new Map<string, string>();
      for (const org of orgs) {
        const res = await fetch(
          `${githubApiUrl}/orgs/${encodeURIComponent(org)}/repos?type=public&per_page=100`,
        );
        if (!res.ok) continue;
        const repos: OrgRepo[] = await res.json();
        for (const repo of repos) {
          if (repo.language) {
            languages.set(repo.full_name.toLowerCase(), repo.language);
          }
        }
      }
      return languages;
    } catch {
      return null;
    }
  },
);

const getRecentCommits = cached(
  async (): Promise<CommitSearchItem[] | null> => {
    const items: CommitSearchItem[] = [];
    for (let page = 1; page <= MAX_COMMIT_PAGES; page++) {
      try {
        const res = await fetch(
          `${githubApiUrl}/search/commits?q=author:${userData.githubUser}&sort=committer-date&order=desc&per_page=100&page=${page}`,
        );
        if (!res.ok) break;
        const data = await res.json();
        const batch: CommitSearchItem[] = data.items ?? [];
        items.push(...batch);
        if (batch.length < 100) break;
      } catch {
        break;
      }
    }
    return items.length > 0 ? items : null;
  },
);

function buildProfileTitle(login: string, name: string | null): string {
  if (!name) return login;
  const normalizedName = name.replace(/\s+/g, " ").trim();
  return normalizedName ? `${login} (${normalizedName})` : login;
}

function toMonthlyCounts(days: ContributionData[]) {
  const buckets = new Map<string, number>();
  for (const day of days) {
    const [year, month] = day.date.split("-");
    if (!year || !month) continue;
    const label = `${year.slice(2)}/${month}`;
    buckets.set(label, (buckets.get(label) ?? 0) + Number(day.count));
  }
  return Array.from(buckets, ([label, value]) => ({ label, value }));
}

function topLanguages(counts: Map<string, number>): LanguageStat[] {
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, TOP_LANGUAGES)
    .map(([name, value]) => ({
      name,
      value,
      color: languageColor(name) ?? "#586e75",
    }));
}

export async function fetchProfileStats(): Promise<ProfileStats | null> {
  try {
    const [profile, repos, days] = await Promise.all([
      getProfile(),
      getOwnedRepos(),
      fetchContributionCounts(),
    ]);
    if (!profile) return null;

    const contributions = days
      ? days.reduce((sum, day) => sum + Number(day.count), 0)
      : null;
    const publicRepos = repos
      ? repos.filter((repo) => !repo.fork).length
      : profile.public_repos ?? 0;

    return {
      title: buildProfileTitle(profile.login, profile.name),
      contributions,
      publicRepos,
      createdAt: profile.created_at ?? null,
      email: profile.email ?? null,
      company: profile.company ?? null,
      location: profile.location ?? null,
      monthly: days ? toMonthlyCounts(days) : [],
    };
  } catch {
    return null;
  }
}

export async function fetchRepoLanguages(): Promise<LanguageStat[] | null> {
  try {
    const repos = await getOwnedRepos();
    if (!repos) return null;
    const counts = new Map<string, number>();
    for (const repo of repos) {
      if (repo.fork || !repo.language) continue;
      counts.set(repo.language, (counts.get(repo.language) ?? 0) + 1);
    }
    return topLanguages(counts);
  } catch {
    return null;
  }
}

export async function fetchCommitLanguages(): Promise<LanguageStat[] | null> {
  try {
    const [commits, ownedRepos, orgLanguages] = await Promise.all([
      getRecentCommits(),
      getOwnedRepos(),
      getOrgRepoLanguages(),
    ]);
    if (!commits || !ownedRepos) return null;

    const languageByRepo = new Map<string, string>();
    for (const repo of ownedRepos) {
      if (repo.language) {
        languageByRepo.set(repo.full_name.toLowerCase(), repo.language);
      }
    }
    if (orgLanguages) {
      orgLanguages.forEach((language, fullName) => {
        languageByRepo.set(fullName, language);
      });
    }

    const counts = new Map<string, number>();
    for (const commit of commits) {
      const language = languageByRepo.get(
        commit.repository.full_name.toLowerCase(),
      );
      if (language) {
        counts.set(language, (counts.get(language) ?? 0) + 1);
      }
    }
    return topLanguages(counts);
  } catch {
    return null;
  }
}

export async function fetchProductiveTime(): Promise<number[] | null> {
  try {
    const commits = await getRecentCommits();
    if (!commits) return null;

    const parsedOffset = Number(userData.timezone);
    const clampedOffset = Number.isFinite(parsedOffset)
      ? Math.min(14, Math.max(-12, parsedOffset))
      : 0;
    const offsetHours = Math.floor(clampedOffset);
    const since = Date.now() - 365 * DAY_MS;

    const buckets = new Array<number>(24).fill(0);
    let sampled = 0;
    for (const commit of commits) {
      const date = new Date(commit.commit.author.date);
      const time = date.getTime();
      if (Number.isNaN(time) || time < since) continue;
      const hour = date.getUTCHours() + offsetHours;
      buckets[((hour % 24) + 24) % 24] += 1;
      sampled += 1;
    }
    return sampled > 0 ? buckets : null;
  } catch {
    return null;
  }
}

function userToday(): string {
  const parsedOffset = Number(userData.timezone);
  const offset = Number.isFinite(parsedOffset)
    ? Math.min(14, Math.max(-12, parsedOffset))
    : 0;
  const shifted = new Date(Date.now() + offset * 60 * 60 * 1000);
  return shifted.toISOString().slice(0, 10);
}

function computeStreakStats(
  days: { date: string; count: number }[],
  today: string,
): StreakStats | null {
  const sorted = [...days].sort((a, b) =>
    a.date < b.date ? -1 : a.date > b.date ? 1 : 0,
  );
  const first = sorted[0].date;
  const current: StreakRange = { length: 0, start: first, end: first };
  const longest: StreakRange = { length: 0, start: first, end: first };
  let total = 0;
  let firstContribution = "";

  for (const day of sorted) {
    total += day.count;
    if (day.count > 0) {
      current.length += 1;
      current.end = day.date;
      if (current.length === 1) current.start = day.date;
      if (!firstContribution) firstContribution = day.date;
      if (current.length > longest.length) {
        longest.start = current.start;
        longest.end = current.end;
        longest.length = current.length;
      }
    } else if (day.date !== today) {
      current.length = 0;
      current.start = today;
      current.end = today;
    }
  }

  if (!firstContribution) return null;
  return { total, firstContribution, current, longest, today };
}

const getStreakStats = cached(async (): Promise<StreakStats | null> => {
  try {
    const profile = await getProfile();
    if (!profile || !profile.created_at) return null;

    const today = userToday();
    const currentYear = Number(today.slice(0, 4));
    const createdYear = Number(profile.created_at.slice(0, 4));
    const startYear =
      Number.isFinite(createdYear) && createdYear >= MIN_CONTRIBUTION_YEAR
        ? Math.min(createdYear, currentYear)
        : currentYear;

    const res = await fetch(
      `${contributionApiUrl}?user=${userData.githubUser}&from=${startYear}&to=${currentYear}`,
    );
    if (!res.ok) return null;
    const data = (await res.json()) as {
      total: number;
      contributions: ContributionData[][];
    };

    const days = (data.contributions ?? [])
      .flat()
      .filter((day) => day.date <= today)
      .map((day) => ({ date: day.date, count: Number(day.count) }));
    if (days.length === 0) return null;
    return computeStreakStats(days, today);
  } catch {
    return null;
  }
});

export function fetchStreakStats(): Promise<StreakStats | null> {
  return getStreakStats();
}
