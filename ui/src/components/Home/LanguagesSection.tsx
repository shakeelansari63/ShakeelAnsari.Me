import { Card } from "primereact/card";
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
                        <img
                            src={getTopLanguageByRepo()}
                            className="max-w-full"
                            alt="Languages by repo"
                        />
                    </Card>
                </div>
                <div className="md:col-6 col-12">
                    <Card className="text-center">
                        <img
                            src={getTopLanguageByCommit()}
                            className="max-w-full"
                            alt="Languages by commit"
                        />
                    </Card>
                </div>
            </div>
        </>
    );
}
