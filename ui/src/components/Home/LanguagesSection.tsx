import { Card } from "primereact/card";
import LazyImage from "../shared/LazyImage";
import {
    getTopLanguageByRepo,
    getTopLanguageByCommit,
} from "../../services/stats";

export default function LanguagesSection() {
    return (
        <>
            <div className="grid">
                <div className="md:col-6 col-12">
                    <Card className="text-center">
                        <LazyImage
                            src={getTopLanguageByRepo()}
                            alt="Languages by repo"
                        />
                    </Card>
                </div>
                <div className="md:col-6 col-12">
                    <Card className="text-center">
                        <LazyImage
                            src={getTopLanguageByCommit()}
                            alt="Languages by commit"
                        />
                    </Card>
                </div>
            </div>
        </>
    );
}
