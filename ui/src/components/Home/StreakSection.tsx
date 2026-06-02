import { Card } from "primereact/card";
import LazyImage from "../shared/LazyImage";
import { getStreaks, getProductiveTime } from "../../services/stats";

export default function StreakSection() {
    return (
        <div className="grid">
            <div className="md:col-6 col-12">
                <Card
                    className="text-center h-full"
                    pt={{
                        content: {
                            className:
                                "flex align-items-center justify-content-center",
                        },
                    }}
                >
                    <LazyImage src={getStreaks()} alt="Streaks" />
                </Card>
            </div>
            <div className="md:col-6 col-12">
                <Card
                    className="text-center h-full"
                    pt={{
                        content: {
                            className:
                                "flex align-items-center justify-content-center",
                        },
                    }}
                >
                    <LazyImage
                        src={getProductiveTime()}
                        alt="Productive time"
                    />
                </Card>
            </div>
        </div>
    );
}
