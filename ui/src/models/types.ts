export interface GitProfile {
  avatar_url: string | null;
  bio: string | null;
  company: string | null;
  location: string | null;
  followers: number | null;
  following: number | null;
  public_repos: number | null;
  hireable: string | null;
  name: string | null;
  login: string;
  html_url: string | null;
  email?: string | null;
  created_at?: string | null;
}

export interface GitProject {
  id: number | null;
  name: string | null;
  description: string | null;
  html_url: string | null;
  stargazers_count: number | null;
  language: string | null;
  fork: boolean | null;
}

export interface HeatMapDate {
  date: Date;
  value: number | null;
}

export interface CellData extends HeatMapDate {
  cssClass: string;
}

export interface ContributionData {
  date: string;
  intensity: string;
  count: string;
}

export interface ContributionResult {
  total: number;
  contributions: ContributionData[][];
}

export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  content?: string;
  date: string;
  readTime: string;
  tags: string[];
  bannerImage?: string;
  views: number;
  likes: number;
}

export interface ContribSubject {
  startDate: Date;
  endDate: Date;
  data: HeatMapDate[];
}

export interface LearnSubject {
  id: string;
  title: string;
  folder: string;
  sort_order: number;
  thumbnail: string;
}

export interface AnalyticsData {
    viewsByDate: { date: string; count: number }[];
    likesByDate: { date: string; count: number }[];
    viewsByCountry: { code: string; country: string; count: number }[];
    likesByCountry: { code: string; country: string; count: number }[];
    topBlogs: { id: string; title: string; views: number; likes: number }[];
    totalViews: number;
    totalLikes: number;
    uniqueVisitors: number;
}

export interface LearnChapter {
  id: number;
  chapter_id: string;
  title: string;
  sort_order: number;
}

export interface LanguageStat {
  name: string;
  value: number;
  color: string;
}

export interface MonthlyContribution {
  label: string;
  value: number;
}

export interface ProfileStats {
  title: string;
  contributions: number | null;
  publicRepos: number;
  createdAt: string | null;
  email: string | null;
  company: string | null;
  location: string | null;
  monthly: MonthlyContribution[];
}

export interface StreakRange {
  length: number;
  start: string;
  end: string;
}

export interface StreakStats {
  total: number;
  firstContribution: string;
  current: StreakRange;
  longest: StreakRange;
  today: string;
}
