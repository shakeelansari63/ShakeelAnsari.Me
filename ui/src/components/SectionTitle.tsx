interface SectionTitleProps {
  anchor?: string;
  hideTitle?: boolean;
  children?: React.ReactNode;
}

export default function SectionTitle({ anchor, hideTitle, children }: SectionTitleProps) {
  return (
    <>
      <div className="mb-6 w-full h-2rem" id={anchor} />
      {!hideTitle && (
        <div className="flex justify-content-center align-items-center mb-4">
          <div className="section-title-left" />
          <div className="section-title-box flex border-round text-white px-4 py-1 shadow-4 font-bold">
            {children}
          </div>
          <div className="section-title-right" />
        </div>
      )}
    </>
  );
}
