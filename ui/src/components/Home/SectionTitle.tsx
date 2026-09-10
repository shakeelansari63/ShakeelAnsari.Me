import { Divider, Typography } from 'antd';

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
      <div className="mb-6 w-full h-5" id={anchor} />
      {!hideTitle && (
        <Divider orientation="left" style={{ marginBottom: 16 }}>
          <Typography.Text strong style={{ color: 'var(--ant-color-primary)' }}>
            {children}
          </Typography.Text>
        </Divider>
      )}
    </>
  );
}