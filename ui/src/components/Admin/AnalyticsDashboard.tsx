import { useState, useEffect, useRef } from "react";
import { Dropdown } from "primereact/dropdown";
import { Toast } from "primereact/toast";
import type { AnalyticsData } from "../../models/types";
import { fetchAnalytics } from "../../services/api";
import SummaryCards from "./SummaryCards";
import DailyBarChart from "./DailyBarChart";
import CountryPieChart from "./CountryPieChart";
import TopBlogsList from "./TopBlogsList";

interface Props {
    token: string;
}

export default function AnalyticsDashboard({ token }: Props) {
    const toast = useRef<Toast>(null);
    const [data, setData] = useState<AnalyticsData | null>(null);
    const [loading, setLoading] = useState(false);
    const [selectedBlog, setSelectedBlog] = useState<string | null>(null);

    const blogOptions = data?.topBlogs.map((b) => ({
        label: b.title,
        value: b.id,
    })) ?? [];

    const load = (blogId?: string) => {
        setLoading(true);
        fetchAnalytics(token, blogId)
            .then((result) => {
                if (result) {
                    setData(result);
                } else {
                    toast.current?.show({ severity: 'error', summary: 'Error', detail: 'Failed to load analytics', life: 3000 });
                }
            })
            .finally(() => setLoading(false));
    };

    useEffect(() => {
        load(selectedBlog ?? undefined);
    }, [selectedBlog]);

    return (
        <>
            <Toast ref={toast} />
            <div style={{ maxWidth: '400px' }} className="mb-3">
                <Dropdown
                    value={selectedBlog}
                    options={[{ label: 'All Blogs', value: null }, ...blogOptions]}
                    onChange={(e) => setSelectedBlog(e.value)}
                    placeholder="All Blogs"
                    className="w-full"
                    style={{ outline: 'none', boxShadow: 'none' }}
                />
            </div>
            {loading ? (
                <p className="text-gray-400">Loading analytics...</p>
            ) : data ? (
                <div className="flex flex-column gap-4">
                    <SummaryCards
                        totalViews={data.totalViews}
                        totalLikes={data.totalLikes}
                        uniqueVisitors={data.uniqueVisitors}
                    />

                    <DailyBarChart title="Daily Views" data={data.viewsByDate} color="#d53a9d" />
                    <DailyBarChart title="Daily Likes" data={data.likesByDate} color="#22c55e" />

                    <div className="grid">
                        <div className="md:col-6 col-12">
                            <CountryPieChart title="Views by Country" data={data.viewsByCountry} />
                        </div>
                        <div className="md:col-6 col-12">
                            <CountryPieChart title="Likes by Country" data={data.likesByCountry} />
                        </div>
                    </div>

                    <TopBlogsList blogs={data.topBlogs} />
                </div>
            ) : (
                <p className="text-gray-400">Failed to load analytics.</p>
            )}
        </>
    );
}