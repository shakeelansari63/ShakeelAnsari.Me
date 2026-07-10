import { Toolbar } from "primereact/toolbar";
import { Button } from "primereact/button";
import { seo } from "../../data/seo";
import { userData } from "../../data/profile";

interface Props {
    isLight?: boolean;
}

export default function PageFooter({ isLight }: Props) {
    const startContent = (
        <span className="text-sm text-gray-400">
            <strong>&copy; {seo.domain}</strong>
        </span>
    );

    const repoUrl = "https://github.com/shakeelansari63/ShakeelAnsari.Me";

    const endContent = (
        <div className="flex gap-2">
            <Button
                text
                severity="secondary"
                size="small"
                icon="pi pi-star"
                label="Star"
                className={isLight ? "" : "text-pink-500"}
                onClick={() => window.open(repoUrl, "_blank")}
                style={{ outline: "none", boxShadow: "none" }}
            />
            <a href={userData.github} target="_blank" title="GitHub">
                <Button
                    icon="pi pi-github"
                    text
                    severity="secondary"
                    size="small"
                    className={isLight ? "" : "text-pink-500"}
                    style={{ outline: "none", boxShadow: "none" }}
                />
            </a>
            <a href={userData.linkedIn} target="_blank" title="LinkedIn">
                <Button
                    icon="pi pi-linkedin"
                    text
                    severity="secondary"
                    size="small"
                    className={isLight ? "" : "text-pink-500"}
                    style={{ outline: "none", boxShadow: "none" }}
                />
            </a>
            <a href={userData.twitter} target="_blank" title="X">
                <Button
                    icon="pi pi-twitter"
                    text
                    severity="secondary"
                    size="small"
                    className={isLight ? "" : "text-pink-500"}
                    style={{ outline: "none", boxShadow: "none" }}
                />
            </a>
        </div>
    );

    return (
        <Toolbar
            start={startContent}
            end={endContent}
            className="border-none bg-transparent mt-4"
        />
    );
}
