import { Tag, Space } from 'antd';
import { userData } from '../../data/profile';
import { settings } from '../../data/settings';

export default function SkillsSection() {
  const tagStyle = {
    backgroundColor: `${settings.themeColor}1a`,
    borderColor: `${settings.themeColor}4d`,
    color: settings.themeColor,
    fontSize: '1rem',
    padding: '6px 16px',
    borderRadius: '999px',
    fontWeight: 500,
    transition: 'all 0.2s',
    cursor: 'default',
  };

  return (
    <Space wrap size={[8, 8]} style={{ width: '100%', justifyContent: 'center' }}>
      {userData.skills.map(skill => (
        <Tag
          key={skill}
          style={tagStyle}
          onMouseEnter={e => {
            e.currentTarget.style.transform = 'scale(1.05)';
            e.currentTarget.style.boxShadow = `0 0 12px ${settings.themeColor}66`;
          }}
          onMouseLeave={e => {
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          {skill}
        </Tag>
      ))}
    </Space>
  );
}