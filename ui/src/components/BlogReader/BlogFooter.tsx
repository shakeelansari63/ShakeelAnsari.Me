import { Toolbar } from "primereact/toolbar";
import { Button } from "primereact/button";
import { userData } from "../../services/data";

interface Props {
    isLight?: boolean;
}

export default function BlogFooter({ isLight }: Props) {
    const startContent = (
        <span className="text-sm">
            <span className="mr-1">Editor credit by</span>
            <strong>@{userData.githubUser}</strong>
        </span>
    );

    const endContent = (
        <div className="flex gap-2">
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
