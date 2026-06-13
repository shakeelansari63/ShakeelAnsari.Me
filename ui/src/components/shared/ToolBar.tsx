import { Suspense, lazy, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Toolbar } from "primereact/toolbar";
import { Button } from "primereact/button";
import { userData } from "../../data/profile";
import { settings } from "../../data/settings";

const Sidebar = lazy(() =>
    import("primereact/sidebar").then((m) => ({ default: m.Sidebar }))
);

interface Props {
    isLight?: boolean;
    onToggleTheme?: () => void;
}

export default function ToolBar({ isLight, onToggleTheme }: Props) {
    const navigate = useNavigate();
    const location = useLocation();
    const [menuOpen, setMenuOpen] = useState(false);

    const scrollTo = (id: string) => {
        const el = document.getElementById(id);
        if (el) {
            el.scrollIntoView({ behavior: "smooth" });
        }
    };

    const homeNav = (id: string) => {
        setMenuOpen(false);
        if (location.pathname !== "/") {
            navigate("/");
            setTimeout(() => scrollTo(id), 300);
        } else {
            scrollTo(id);
        }
    };

    const menuItems = [
        ...(settings.showExpo
            ? [
                  {
                      label: "Expo",
                      icon: "pi pi-briefcase",
                      visible: !location.pathname.startsWith("/expo"),
                      action: () => {
                          setMenuOpen(false);
                          navigate("/expo");
                      },
                  },
              ]
            : []),
        ...(settings.showBlogs
            ? [
                  {
                      label: "Blogs",
                      icon: "pi pi-book",
                      visible: !location.pathname.startsWith("/blog"),
                      action: () => {
                          setMenuOpen(false);
                          navigate("/blog");
                      },
                  },
              ]
            : []),
        ...(settings.showTutorial
            ? [
                  {
                      label: "Learn",
                      icon: "pi pi-graduation-cap",
                      visible: !location.pathname.startsWith("/learn"),
                      action: () => {
                          setMenuOpen(false);
                          navigate("/learn");
                      },
                  },
              ]
            : []),
        {
            label: "Stats",
            icon: "pi pi-chart-bar",
            visible: location.pathname === "/",
            action: () => homeNav("stats"),
        },
        {
            label: "Contributions",
            icon: "pi pi-calendar",
            visible: location.pathname === "/",
            action: () => homeNav("contributions"),
        },
        {
            label: "Projects",
            icon: "pi pi-folder",
            visible: location.pathname === "/",
            action: () => homeNav("projects"),
        },
    ];

    const startContent =
        location.pathname === "/" ? (
            <span
                className="text-pink-500 font-bold text-2xl no-underline cursor-pointer"
                onClick={() => navigate("/")}
            >
                @{userData.devUsername}
            </span>
        ) : (
            <Button
                text
                severity="secondary"
                className="text-pink-500 no-hover-bg"
                icon="pi pi-home"
                label="Home"
                onClick={() => navigate("/")}
                style={{ outline: "none", boxShadow: "none" }}
            />
        );

    const endContent = (
        <>
            <div className="hidden md:flex gap-2">
                {settings.showExpo && !location.pathname.startsWith("/expo") && (
                    <Button
                        text
                        severity="secondary"
                        className="text-pink-500"
                        icon="pi pi-briefcase"
                        label="Expo"
                        onClick={() => navigate("/expo")}
                        style={{ outline: "none", boxShadow: "none" }}
                    />
                )}
                {settings.showBlogs && !location.pathname.startsWith("/blog") && (
                    <Button
                        text
                        severity="secondary"
                        className="text-pink-500"
                        icon="pi pi-book"
                        label="Blogs"
                        onClick={() => navigate("/blog")}
                        style={{ outline: "none", boxShadow: "none" }}
                    />
                )}
                {settings.showTutorial && !location.pathname.startsWith("/learn") && (
                    <Button
                        text
                        severity="secondary"
                        className="text-pink-500"
                        icon="pi pi-graduation-cap"
                        label="Learn"
                        onClick={() => navigate("/learn")}
                        style={{ outline: "none", boxShadow: "none" }}
                    />
                )}
                {location.pathname === "/" && (
                    <>
                        <Button
                            text
                            severity="secondary"
                            className="text-pink-500"
                            icon="pi pi-chart-bar"
                            label="Stats"
                            onClick={() => scrollTo("stats")}
                            style={{ outline: "none", boxShadow: "none" }}
                        />
                        <Button
                            text
                            severity="secondary"
                            className="text-pink-500"
                            icon="pi pi-calendar"
                            label="Contributions"
                            onClick={() => scrollTo("contributions")}
                            style={{ outline: "none", boxShadow: "none" }}
                        />
                        <Button
                            text
                            severity="secondary"
                            className="text-pink-500"
                            icon="pi pi-folder"
                            label="Projects"
                            onClick={() => scrollTo("projects")}
                            style={{ outline: "none", boxShadow: "none" }}
                        />
                    </>
                )}
            </div>
            {onToggleTheme && (
                <Button
                    text
                    severity="secondary"
                    className="text-pink-500"
                    icon={isLight ? "pi pi-moon" : "pi pi-sun"}
                    onClick={onToggleTheme}
                    tooltip={isLight ? "Dark Mode" : "Light Mode"}
                    tooltipOptions={{ position: "bottom" }}
                    style={{ outline: "none", boxShadow: "none" }}
                />
            )}
            <div className="md:hidden">
                <Button
                    icon="pi pi-bars"
                    text
                    severity="secondary"
                    className="text-pink-500"
                    onClick={() => setMenuOpen(true)}
                    style={{ outline: "none", boxShadow: "none" }}
                />
            </div>
        </>
    );

    return (
        <>
            <Suspense fallback={null}>
                <Sidebar
                    visible={menuOpen}
                    onHide={() => setMenuOpen(false)}
                    position="right"
                    style={{
                        background: "#18181b",
                        border: "none",
                        width: "220px",
                    }}
                >
                    <div className="flex flex-column gap-2 mt-4">
                        {menuItems
                            .filter((item) => item.visible)
                            .map((item) => (
                                <Button
                                    key={item.label}
                                    text
                                    severity="secondary"
                                    className="text-pink-500"
                                    icon={item.icon}
                                    label={item.label}
                                    onClick={item.action}
                                    style={{
                                        outline: "none",
                                        boxShadow: "none",
                                        justifyContent: "flex-start",
                                        width: "100%",
                                    }}
                                    pt={{
                                        label: {
                                            style: {
                                                flex: "none",
                                                textAlign: "left" as const,
                                            },
                                        },
                                    }}
                                />
                            ))}
                    </div>
                </Sidebar>
            </Suspense>
            <Toolbar
                start={startContent}
                end={endContent}
                className="border-none"
                style={{
                    position: "sticky",
                    top: 0,
                    zIndex: 1000,
                    background: "#18181b",
                    borderRadius: "0 0 8px 8px",
                    boxShadow: "0 2px 4px -1px rgba(128, 128, 128, 0.3)",
                }}
            />
        </>
    );
}
