export interface WorkRole {
    title: string;
    startDate: string;
    endDate: string | null;
    description: string;
}

export interface WorkExperience {
    company: string;
    roles: WorkRole[];
}

export const work: WorkExperience[] = [
    {
        company: "UBS",
        roles: [
            {
                title: "Associate Director - Technology",
                startDate: "2022-11",
                endDate: null,
                description:
                    "Working as Data, AI engineer for Wealth Management Americas division.",
            },
        ],
    },
    {
        company: "Teradata",
        roles: [
            {
                title: "Senior Technical Consultant",
                startDate: "2020-02",
                endDate: "2022-11",
                description:
                    "Worked as Senior Technical Consultant for Australian banking client, helping them manage their Analytics platform.",
            },
        ],
    },
    {
        company: "IBM",
        roles: [
            {
                title: "DevOps Engineer",
                startDate: "2018-07",
                endDate: "2020-02",
                description:
                    "Worked as a DevOps Engineer for Australian banking client, maintaining Teradata, Hadoop and Aster platforms.",
            },
            {
                title: "System Administrator",
                startDate: "2018-01",
                endDate: "2018-07",
                description:
                    "Worked as a System Administrator for Australian banking client, maintaining Teradata, Hadoop and Aster platforms.",
            },
            {
                title: "Big Data Engineer",
                startDate: "2016-05",
                endDate: "2018-01",
                description:
                    "Worked as an Analyst Programmer for Australian banking client, maintaining Teradata, Hadoop and Aster platforms.",
            },
            {
                title: "ETL developer",
                startDate: "2012-11",
                endDate: "2016-05",
                description:
                    "Worked as ETL developer in US Healthcare firm's Data warehouse.",
            },
        ],
    },
];
