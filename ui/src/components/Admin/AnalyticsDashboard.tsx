import type { AnalyticsData } from "../../models/types";
import SummaryCards from "./SummaryCards";
import DailyBarChart from "./DailyBarChart";
import CountryPieChart from "./CountryPieChart";
import TopBlogsList from "./TopBlogsList";

interface Props {
    data: AnalyticsData;
}

export default function AnalyticsDashboard({ data }: Props) {
    return (
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
    );
}
