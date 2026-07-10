import { userData } from "../../data/profile";

export default function SkillsSection() {
    return (
        <div className="flex gap-3 flex-wrap justify-content-center">
            {userData.skills.map((skill) => (
                <span
                    key={skill}
                    className="skill-pill"
                    style={{
                        background: "#d53a9d",
                        color: "#fff",
                        padding: "0.375rem 1rem",
                        fontSize: "1.1rem",
                        borderRadius: "999px",
                        lineHeight: 1.5,
                        whiteSpace: "nowrap",
                    }}
                >
                    {skill}
                </span>
            ))}
        </div>
    );
}
