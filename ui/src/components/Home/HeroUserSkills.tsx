import { Card } from "primereact/card";
import type { GitProfile } from "../../models/types";
import { userData } from "../../services/data";

interface Props {
    profile: GitProfile | null;
}

const codeHeader = (
    <div className="code-header">
        <span className="dot-red" />
        <span className="dot-orange" />
        <span className="dot-green" />
    </div>
);

export default function HeroUserSkills({ profile }: Props) {
    return (
        <Card header={codeHeader} className="h-full">
            <div
                className="text-lg overflow-auto"
                style={{
                    wordBreak: "break-word",
                    fontFamily: "SpaceMono, Courier New, monospace",
                    lineHeight: 1.5,
                }}
            >
                <div className="flex flex-row">
                    <span className="text-pink-500 px-1">const</span>
                    <span className="text-purple-500 font-bold px-1">
                        {userData?.alias ?? "coder"}
                    </span>
                    <span className="text-pink-500 px-1">=</span>
                    <span className="px-1">{"{"}</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">name :</span>
                    <span className="px-1 text-green-500">
                        &quot;{profile?.name ?? userData.devUsername}&quot;
                    </span>
                    <span className="px-1 min-w-max">,</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">company :</span>
                    <span className="px-1 text-green-500">
                        &quot;{profile?.company ?? ""}&quot;
                    </span>
                    <span className="px-1 min-w-max">,</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">location :</span>
                    <span className="px-1 text-green-500">
                        &quot;{profile?.location ?? ""}&quot;
                    </span>
                    <span className="px-1 min-w-max">,</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">followers :</span>
                    <span className="px-1 text-orange-500">
                        {profile?.followers ?? 0}
                    </span>
                    <span className="px-1 min-w-max">,</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">following :</span>
                    <span className="px-1 text-orange-500">
                        {profile?.following ?? 0}
                    </span>
                    <span className="px-1 min-w-max">,</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">repositories :</span>
                    <span className="px-1 text-orange-500">
                        {profile?.public_repos ?? 0}
                    </span>
                    <span className="px-1 min-w-max">,</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">hirable :</span>
                    <span className="px-1 text-orange-500">
                        {String(profile?.hireable ?? false)}
                    </span>
                    <span className="px-1 min-w-max">,</span>
                </div>
                <div className="flex flex-row">
                    <span className="pl-6" />
                    <span className="px-1 min-w-max">skills :</span>
                    <span className="px-1 flex flex-wrap">
                        [&nbsp;
                        {userData.skills.map((skill, i) => (
                            <span key={skill} className="flex-nowrap">
                                <span className="text-blue-400">
                                    &quot;{skill}&quot;
                                </span>
                                {i < userData.skills.length - 1 && (
                                    <span className="text-white">,&nbsp;</span>
                                )}
                            </span>
                        ))}
                        &nbsp;],
                    </span>
                </div>
                <div className="flex flex-row">
                    <span className="px-1">{"};"}</span>
                </div>
            </div>
        </Card>
    );
}
