import { Card } from "primereact/card";
import { Button } from "primereact/button";
import LazyImage from "../shared/LazyImage";
import type { GitProfile } from "../../models/types";
import { userData } from "../../services/data";

interface Props {
    profile: GitProfile | null;
}

export default function HeroUserDetail({ profile }: Props) {
    const footer = (
        <div className="flex flex-column align-items-center">
            <div className="flex align-items-center justify-content-center flex-wrap gap-3">
                <a href={userData.github} target="_blank" title="GitHub">
                    <Button
                        icon="pi pi-github"
                        text
                        rounded
                        severity="secondary"
                        className="text-xl text-pink-500"
                        style={{ outline: "none", boxShadow: "none" }}
                    />
                </a>
                <a href={userData.email} title="Email">
                    <Button
                        icon="pi pi-envelope"
                        text
                        rounded
                        severity="secondary"
                        className="text-xl text-pink-500"
                        style={{ outline: "none", boxShadow: "none" }}
                    />
                </a>
                <a href={userData.linkedIn} target="_blank" title="LinkedIn">
                    <Button
                        icon="pi pi-linkedin"
                        text
                        rounded
                        severity="secondary"
                        className="text-xl text-pink-500"
                        style={{ outline: "none", boxShadow: "none" }}
                    />
                </a>
                <a href={userData.twitter} target="_blank" title="Twitter">
                    <Button
                        icon="pi pi-twitter"
                        text
                        rounded
                        severity="secondary"
                        className="text-xl text-pink-500"
                        style={{ outline: "none", boxShadow: "none" }}
                    />
                </a>
            </div>
            <div className="mt-3">
                <Button
                    label="Check my badges"
                    icon="pi pi-verified"
                    rounded
                    className="border-gradient-purple text-white"
                    onClick={() => window.open(userData.badges, "_blank")}
                    style={{ outline: "none", boxShadow: "none" }}
                />
            </div>
        </div>
    );

    return (
        <Card footer={footer} className="h-full">
            <div className="flex flex-column align-items-center p-3">
                <LazyImage
                    src={profile?.avatar_url ?? ""}
                    alt="Avatar"
                    maxWidth={180}
                    maxHeight={180}
                    aspectRatio="1 / 1"
                    rounded
                />
                <div className="flex align-items-center justify-content-center mb-2 text-center">
                    <span className="text-orange-300 text-lg">
                        {profile?.bio ? `⚡${profile.bio}⚡` : ""}
                    </span>
                </div>
            </div>
        </Card>
    );
}
