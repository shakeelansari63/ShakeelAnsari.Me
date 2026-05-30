interface SectionTitleProps {
    anchor?: string;
    hideTitle?: boolean;
    children?: React.ReactNode;
}

export default function SectionTitle({
    anchor,
    hideTitle,
    children,
}: SectionTitleProps) {
    return (
        <>
            <div className="mb-6 w-full h-2rem" id={anchor} />
            {!hideTitle && (
                <div className="flex align-items-center mb-4">
                    <div
                        className="flex-1"
                        style={{
                            height: "1px",
                            background:
                                "linear-gradient(to right, transparent, #271250)",
                        }}
                    />
                    <span
                        className="font-bold text-white px-4 py-1"
                        style={{ background: "#271250", borderRadius: "6px" }}
                    >
                        {children}
                    </span>
                    <div
                        className="flex-1"
                        style={{
                            height: "1px",
                            background:
                                "linear-gradient(to left, transparent, #271250)",
                        }}
                    />
                </div>
            )}
        </>
    );
}
