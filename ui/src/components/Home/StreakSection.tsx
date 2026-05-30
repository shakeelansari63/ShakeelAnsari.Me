import { Card } from "primereact/card";
import { getStreaks, getProductiveTime } from "../../services/stats";

export default function StreakSection() {
    return (
        <div className="grid">
            <div className="md:col-6 col-12">
                <Card className="text-center">
                    <img
                        src={getStreaks()}
                        className="max-w-full"
                        alt="Streaks"
                    />
                </Card>
            </div>
            <div className="md:col-6 col-12">
                <Card className="text-center">
                    <img
                        src={getProductiveTime()}
                        className="max-w-full"
                        alt="Productive time"
                    />
                </Card>
            </div>
        </div>
    );
}
