import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import type { PieLabelRenderProps } from "recharts";
import { Card } from "primereact/card";

const COLORS = ["#d53a9d", "#743ad5", "#22c55e", "#f59e0b", "#3b82f6", "#ef4444", "#14b8a6", "#f97316", "#8b5cf6", "#ec4899"];

interface CountryEntry {
    code: string;
    country: string;
    count: number;
}

interface Props {
    title: string;
    data: CountryEntry[];
}

export default function CountryPieChart({ title, data }: Props) {
    return (
        <Card>
            <h3 className="text-white text-lg font-bold mb-3">{title}</h3>
            <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                    <Pie
                        data={data}
                        dataKey="count"
                        nameKey="country"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label={({ payload }: PieLabelRenderProps) =>
                            `${payload?.country ?? ""} (${payload?.count ?? 0})`
                        }
                    >
                        {data.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                    </Pie>
                    <Tooltip
                        contentStyle={{ background: "#1f2023", border: "1px solid #333", borderRadius: "6px" }}
                        labelStyle={{ color: "#fff" }}
                    />
                </PieChart>
            </ResponsiveContainer>
        </Card>
    );
}
