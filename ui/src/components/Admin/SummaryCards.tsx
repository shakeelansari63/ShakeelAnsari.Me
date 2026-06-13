import { Card } from "primereact/card";

interface Props {
    totalViews: number;
    totalLikes: number;
    uniqueVisitors: number;
}

export default function SummaryCards({ totalViews, totalLikes, uniqueVisitors }: Props) {
    const items = [
        { value: totalViews, label: "Total Views" },
        { value: totalLikes, label: "Total Likes" },
        { value: uniqueVisitors, label: "Unique Visitors" },
    ];

    return (
        <div className="grid">
            {items.map((item) => (
                <div key={item.label} className="md:col-4 col-12">
                    <Card className="text-center">
                        <span className="text-3xl font-bold text-pink-400">{item.value}</span>
                        <p className="text-gray-400 m-0 mt-1">{item.label}</p>
                    </Card>
                </div>
            ))}
        </div>
    );
}
