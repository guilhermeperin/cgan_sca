from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
from sqlalchemy import create_engine, and_, Table, Column, MetaData, ForeignKey
from sqlalchemy import Integer, String, JSON, Boolean, Float, Double, SmallInteger, ARRAY, BigInteger, DateTime, Date
from sqlalchemy import BINARY, LargeBinary, Text, Numeric, DOUBLE_PRECISION, Unicode, Time
from pattern.text.en import singularize, pluralize

Base = declarative_base()


class DatabaseEntry:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return None  # To customize the default behavior for non-existing attributes


class SqlAlchemyWrapper:

    def __init__(self, db_file, db_url=None):
        self.db_file = db_file
        self.sqlalchemy_database_file = f"sqlite:///./{db_file}.db"
        self.sqlalchemy_database_url = db_url  # ex: "postgresql://user:password@postgresserver/db"

        self.engine = create_engine(
            self.sqlalchemy_database_url if db_url is not None else self.sqlalchemy_database_file,
            connect_args={"check_same_thread": False}
        )
        self.session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)()
        # Base.metadata.create_all(bind=self.engine)

    def create_table(self, table_name, columns, relationships=None):

        table = Base.metadata.tables.get(table_name)
        if table is not None:
            return
        # Create a dynamic class for the table
        DynamicTable = type(
            table_name.title(), (Base,),
            {"__tablename__": table_name, "id": Column(Integer, primary_key=True, index=True)}
        )

        for column in columns:
            if "foreign_key_table" in column:
                foreign_key_column = Column(column['type'], ForeignKey(f'{column["foreign_key_table"]}.id'))
                setattr(DynamicTable, column['name'], foreign_key_column)

                # Assuming 'relationships' is a list, add the relationship to the foreign table
                foreign_table = Base.metadata.tables.get(column["foreign_key_table"])
                relationship_property = relationship(f"{table_name}_relationship", back_populates=f"{table_name}")
                setattr(DynamicTable, f"{column['name']}_relationship", relationship_property)
                setattr(foreign_table, f"{table_name}_relationship", relationship_property)
            else:
                if "properties" in column:
                    setattr(DynamicTable, column['name'], Column(column['type'], **column['properties']))
                else:
                    setattr(DynamicTable, column['name'], Column(column['type']))

        # Add relationships to the dynamic class
        # if relationships:
        #     for relationship in relationships:
        #         setattr(DynamicTable, relationship['name'], relationship)

        # Create and commit the table
        try:
            DBSession = sessionmaker(bind=self.engine)
            session = DBSession()
            Base.metadata.create_all(bind=self.engine)
            session.commit()
            session.close()
        except Exception as e:
            print(e)

    def create_association_table(self, table1, table2):
        # Create a dynamic class for the table
        association_table_name = f"{singularize(table1).title()}{singularize(table2).title()}"
        DynamicTable = type(association_table_name, (Base,),
                            {"__tablename__": association_table_name,
                             "id": Column(Integer, primary_key=True, index=True)})

        setattr(DynamicTable, f"{singularize(table1)}_id", Column(Integer, ForeignKey(f"{table1}.id")))
        setattr(DynamicTable, f"{singularize(table2)}_id", Column(Integer, ForeignKey(f"{table2}.id")))
        setattr(DynamicTable, table1, relationship(table1.title(), back_populates=table2))
        setattr(DynamicTable, table2, relationship(table2.title(), back_populates=table1))

        table1_obj = Base.metadata.tables.get(table1)
        setattr(table1_obj, table2, relationship(association_table_name, back_populates=table1))
        table2_obj = Base.metadata.tables.get(table2)
        setattr(table2_obj, table1, relationship(association_table_name, back_populates=table2))

        DBSession = sessionmaker(bind=self.engine)
        session = DBSession()
        Base.metadata.create_all(bind=self.engine)
        session.commit()
        session.close()

    def insert_data(self, table_name, data):
        # session = self.make_session()
        session = self.session

        if isinstance(table_name, list):
            association_table_name = f"{singularize(table_name[0]).title()}{singularize(table_name[1]).title()}"
            table = Base.metadata.tables.get(association_table_name)
        else:
            table = Base.metadata.tables.get(table_name)

        if table is None:
            raise ValueError(f"Table '{table_name}' not found.")

        if isinstance(data, list):
            entries = []
            for d in data:
                entries.append(session.execute(table.insert().values(d).returning(table)).fetchone())
                session.commit()

            result_dict_list = []
            for row in entries:
                row_dict = {column.name: getattr(row, column.name) for column in table.columns}
                result_dict_list.append(DatabaseEntry(**row_dict))

            session.close()

            return result_dict_list

        else:
            entry = session.execute(table.insert().values(data).returning(table)).fetchone()
            session.commit()
            row_dict = {column.name: getattr(entry, column.name) for column in table.columns}
            session.close()

            return DatabaseEntry(**row_dict)

    def select_data(self, table_name, conditions=None):
        # session = self.make_session()
        session = self.session

        # Get the dynamically created table from metadata
        table = Base.metadata.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found.")

        # Construct the combined condition
        combined_condition = and_(*[getattr(table.c, column) == value for column, value in (conditions or [])])

        # Select data from the table based on the combined condition
        query = session.query(table)
        if conditions:
            query = query.filter(combined_condition)

        result = query.all()

        # Convert the result to a list of dictionaries with column names as keys
        result_dict_list = []
        for row in result:
            row_dict = {column.name: getattr(row, column.name) for column in table.columns}
            result_dict_list.append(DatabaseEntry(**row_dict))

        # Close the session
        session.close()

        # if len(result_dict_list) == 1:
        #     return result_dict_list[0]
        if len(result_dict_list) == 0:
            return []
        return result_dict_list

    def select_data_by_id(self, table_name, id):
        # session = self.make_session()
        session = self.session

        # Get the dynamically created table from metadata
        table = Base.metadata.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found.")

        # Select data from the table based on the combined condition
        query = session.query(table).filter(table.c.id == id)

        row = query.first()

        if row is None:
            return row

        # Convert the result to a list of dictionaries with column names as keys
        row_dict = {column.name: getattr(row, column.name) for column in table.columns}

        # Close the session
        session.close()

        return DatabaseEntry(**row_dict)

    def select_data_with_association(self, table_name1, table_name2, conditions_table1=None, conditions_table2=None):
        # session = self.make_session()
        session = self.session

        # Get the dynamically created tables from metadata
        table1 = Base.metadata.tables.get(table_name1)
        table2 = Base.metadata.tables.get(table_name2)

        association_table_name = f"{singularize(table_name1).title()}{singularize(table_name2).title()}"
        association_table = Base.metadata.tables.get(association_table_name)

        if table1 is None or table2 is None:
            raise ValueError("One or more tables not found.")

        if association_table is None:
            association_table_name = f"{singularize(table_name2).title()}{singularize(table_name1).title()}"
            association_table = Base.metadata.tables.get(association_table_name)
            if association_table is None:
                raise ValueError("One or more tables not found.")

        join_condition1 = table1.c.id == getattr(association_table.c, f"{singularize(table_name1)}_id")
        join_condition2 = table2.c.id == getattr(association_table.c, f"{singularize(table_name2)}_id")

        # Construct the combined conditions for each table
        combined_condition_table1 = [getattr(table1.c, column) == value for column, value in (conditions_table1 or [])]
        combined_condition_table2 = [getattr(table2.c, column) == value for column, value in (conditions_table2 or [])]

        query = (
            session.query(table1)
            .join(association_table, join_condition1)
            .join(table2, join_condition2)
            .filter(and_(*combined_condition_table1, *combined_condition_table2))
        )

        result1 = query.all()

        query = (
            session.query(table2)
            .join(association_table, join_condition2)
            .join(table1, join_condition1)
            .filter(and_(*combined_condition_table1, *combined_condition_table2))
        )

        result2 = query.all()

        # Close the session
        session.close()

        results_dict = {
            f'{table_name1}': result1,
            f'{table_name2}': result2
        }

        return DatabaseEntry(**results_dict)

    def select_data_with_relation(self, table_and_relation, conditions_table1=None):
        # session = self.make_session()
        session = self.session

        # Get the dynamically created tables from metadata
        tables = table_and_relation.split(".")
        table1 = Base.metadata.tables.get(pluralize(tables[0]))
        table2 = Base.metadata.tables.get(tables[1])
        association_table_name = f"{singularize(tables[0]).title()}{singularize(tables[1]).title()}"
        association_table = Base.metadata.tables.get(association_table_name)

        if table1 is None or table2 is None:
            raise ValueError("One or more tables not found.")

        if association_table is None:
            association_table_name = f"{singularize(tables[1]).title()}{singularize(tables[0]).title()}"
            association_table = Base.metadata.tables.get(association_table_name)

            if association_table is None:
                raise ValueError("One or more tables not found.")

        join_condition1 = table1.c.id == getattr(association_table.c, f"{singularize(tables[0])}_id")
        join_condition2 = table2.c.id == getattr(association_table.c, f"{singularize(tables[1])}_id")

        # Construct the combined conditions for each table
        combined_condition_table1 = and_(
            *[getattr(table1.c, column) == value for column, value in (conditions_table1 or [])])

        query = (
            session.query(table2)
            .join(association_table, join_condition2)
            .join(table1, join_condition2)
            .filter(combined_condition_table1)
        )

        result = query.all()

        # Close the session
        session.close()

        results_dict = {
            f'{table2}': result,
        }

        return DatabaseEntry(**results_dict)

    def update_data(self, table_name, conditions, values):
        # session = self.make_session()
        session = self.session

        # Get the dynamically created table from metadata
        table = Base.metadata.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found.")

        # Construct the combined condition
        combined_condition = and_(*[getattr(table.c, column) == value for column, value in conditions])

        # Update data in the table based on the combined condition and values
        update_statement = table.update().where(combined_condition).values(values).returning(table)

        # Execute the update statement and fetch the updated entry
        result = session.execute(update_statement)
        row = result.fetchone()

        # Commit the changes
        session.commit()
        session.close()

        if row is not None:
            row_dict = {column.name: getattr(row, column.name) for column in table.columns}
        else:
            row_dict = {}

        # Close the session
        session.close()

        return DatabaseEntry(**row_dict)

    def delete_data(self, table_name, conditions):
        # session = self.make_session()
        session = self.session

        # Get the dynamically created table from metadata
        table = Base.metadata.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found.")

        # Construct the combined condition
        combined_condition = and_(*[getattr(table.c, column) == value for column, value in conditions])

        # Delete data from the table based on the combined condition
        session.execute(table.delete().where(combined_condition))

        # Commit the changes
        session.commit()
        session.close()

    def delete_data_from_association(self, table_name1, table_name2, conditions):
        # session = self.make_session()
        session = self.session

        # Get the dynamically created tables from metadata
        table1 = Base.metadata.tables.get(table_name1)
        table2 = Base.metadata.tables.get(table_name2)

        association_table_name = f"{singularize(table_name1).title()}{singularize(table_name2).title()}"
        association_table = Base.metadata.tables.get(association_table_name)

        if table1 is None or table2 is None:
            raise ValueError("One or more tables not found.")

        if association_table is None:
            association_table_name = f"{singularize(table_name2).title()}{singularize(table_name1).title()}"
            association_table = Base.metadata.tables.get(association_table_name)
            if association_table is None:
                raise ValueError("One or more tables not found.")

        # Construct the combined condition
        combined_condition = and_(*[getattr(association_table.c, column) == value for column, value in conditions])

        # Delete data from the table based on the combined condition
        session.execute(association_table.delete().where(combined_condition))

        # Commit the changes
        session.commit()
        session.close()
