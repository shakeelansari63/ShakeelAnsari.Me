import { Toolbar } from "primereact/toolbar";
import { Button } from "primereact/button";

export default function FooterSection() {
    const startContent = (
        <span className="text-sm">
            <span className="mr-1">GitHub Portfolio by</span>
            <strong>@shakeelansari63</strong>
        </span>
    );

    const endContent = (
        <div className="flex gap-2">
            <Button
                text
                severity="secondary"
                size="small"
                icon="pi pi-star"
                label="Star"
                onClick={() =>
                    window.open(
                        "https://github.com/shakeelansari63/ShakeelAnsari.Me",
                        "_blank",
                    )
                }
                style={{ outline: "none", boxShadow: "none" }}
            />
            <Button
                text
                severity="secondary"
                size="small"
                icon="pi pi-copy"
                label="Fork"
                onClick={() =>
                    window.open(
                        "https://github.com/shakeelansari63/ShakeelAnsari.Me/fork",
                        "_blank",
                    )
                }
                style={{ outline: "none", boxShadow: "none" }}
            />
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
