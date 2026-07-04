import { userData } from "../data/profile";
import type { BlogPost, LearnSubject, LearnChapter, AnalyticsData } from "../models/types";
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
            `${baseApiUrl}/search/repositories?q=user:${userData.githubUser}+fork:false+archived:false&sort=updated&order=desc&per_page=4&type=Repositories`,
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
): Promise<{ likes: number; liked: boolean } | null> {
    try {
        const res = await fetch(`/api/blogs/${id}/like`, { method: "POST" });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

export async function fetchLearnSubjects(): Promise<LearnSubject[]> {
    try {
        const res = await fetch("/api/learn/subjects");
        if (!res.ok) return [];
        const data = await res.json();
        return data.data ?? [];
    } catch {
        return [];
    }
}

export async function fetchSubjectChapters(
    subjectId: string,
): Promise<LearnChapter[]> {
    try {
        const res = await fetch(`/api/learn/subjects/${subjectId}/chapters`);
        if (!res.ok) return [];
        const data = await res.json();
        return data.data ?? [];
    } catch {
        return [];
    }
}

export async function fetchChapterContent(
    subjectId: string,
    chapterId: string,
): Promise<{ title: string; content: string } | null> {
    try {
        const res = await fetch(`/api/learn/subjects/${subjectId}/chapters/${chapterId}/content`);
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

export async function fetchProductContent(
    id: string,
): Promise<{ title: string; excerpt: string; content: string } | null> {
    try {
        const res = await fetch(`/api/products/${id}/content`);
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

export async function fetchAnalytics(token: string, blogId?: string): Promise<AnalyticsData | null> {
    try {
        const body: Record<string, string> = {};
        if (blogId) body.blog_id = blogId;
        const res = await fetch("/api/admin/analytics", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify(body),
        });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
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
