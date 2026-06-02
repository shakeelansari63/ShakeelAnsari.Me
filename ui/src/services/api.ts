import { userData } from "./data";
import type { BlogPost } from "../models/types";
import type {
    GitProfile,
    GitProject,
    ContribSubject,
    ContributionData,
    ContributionResult,
} from "../models/types";

const baseApiUrl = "https://api.github.com";
const contribBaseUrl = "https://gh-calendar.rschristian.dev";

export async function fetchUserProfile(): Promise<GitProfile | null> {
    try {
        const res = await fetch(`${baseApiUrl}/users/${userData.githubUser}`);
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

export async function fetchUserProjects(): Promise<GitProject[]> {
    try {
        const res = await fetch(
            `${baseApiUrl}/search/repositories?q=user:${userData.githubUser}+fork:false+archived:false&sort=updated&order=desc&per_page=6&type=Repositories`,
        );
        if (!res.ok) return [];
        const data = await res.json();
        return data.items ?? [];
    } catch {
        return [];
    }
}

export interface BlogPostsResponse {
    data: BlogPost[];
    page: number;
    limit: number;
    total: number;
    totalPages: number;
}

export async function fetchBlogPosts(
    page = 1,
    limit = 5,
): Promise<BlogPostsResponse> {
    try {
        const res = await fetch(`/api/blogs?page=${page}&limit=${limit}`);
        if (!res.ok) {
            return { data: [], page: 1, limit, total: 0, totalPages: 1 };
        }
        return res.json();
    } catch {
        return { data: [], page: 1, limit, total: 0, totalPages: 1 };
    }
}

export async function fetchBlogContent(id: string): Promise<string | null> {
    try {
        const res = await fetch(`/api/blogs/${id}/content`);
        if (!res.ok) return null;
        const data = await res.json();
        return data.content ?? null;
    } catch {
        return null;
    }
}

export async function fetchBlogPost(id: string): Promise<BlogPost | null> {
    try {
        const res = await fetch(`/api/blogs/${id}`);
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

export interface BlogStats {
    views: number;
    likes: number;
    liked?: boolean;
}

export async function fetchBlogStats(id: string): Promise<BlogStats> {
    try {
        const res = await fetch(`/api/blogs/${id}/stats`);
        if (!res.ok) return { views: 0, likes: 0 };
        return res.json();
    } catch {
        return { views: 0, likes: 0 };
    }
}

export async function recordBlogView(
    id: string,
): Promise<{ views: number } | null> {
    try {
        const res = await fetch(`/api/blogs/${id}/view`, { method: "POST" });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

export async function likeBlog(
    id: string,
): Promise<{ likes: number; liked: boolean }> {
    const res = await fetch(`/api/blogs/${id}/like`, { method: "POST" });
    return res.json();
}

export async function fetchUserContributions(): Promise<ContribSubject | null> {
    try {
        const res = await fetch(
            `${contribBaseUrl}/user/${userData.githubUser}`,
        );
        if (!res.ok) return null;
        const contribResult: ContributionResult = await res.json();

        let contribData: ContributionData[] = [];
        contribResult.contributions.forEach(
            (contrib) => (contribData = [...contribData, ...contrib]),
        );

        const hmData = contribData.map((contrib) => ({
            date: new Date(contrib.date),
            value: parseInt(contrib.intensity),
        }));

        const startDate = hmData.reduce((a, b) =>
            a.date < b.date ? a : b,
        ).date;
        const endDate = hmData.reduce((a, b) => (a.date > b.date ? a : b)).date;

        return { data: hmData, startDate, endDate };
    } catch {
        return null;
    }
}
