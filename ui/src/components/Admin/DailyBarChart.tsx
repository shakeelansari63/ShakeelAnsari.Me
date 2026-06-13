import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Card } from "primereact/card";

interface Props {
    title: string;
    data: { date: string; count: number }[];
    color: string;
}

export default function DailyBarChart({ title, data, color }: Props) {
    return (
        <Card>
            <h3 className="text-white text-lg font-bold mb-3">{title}</h3>
            <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="date" stroke="#888" fontSize={12} />
                    <YAxis stroke="#888" fontSize={12} />
                    <Tooltip
                        contentStyle={{ background: "#1f2023", border: "1px solid #333", borderRadius: "6px" }}
                        labelStyle={{ color: "#fff" }}
                    />
                    <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </Card>
    );
}
