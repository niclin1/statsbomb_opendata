import pandas as pd
from statsbombpy import sb
import json

def extract_examples():
    output_md = "docs/statsbomb_data_examples.md"
    
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Exemplos das Bases de Dados da StatsBomb\n\n")
        f.write("Abaixo estão exemplos do formato bruto dos dados extraídos da API da StatsBomb para ilustrar ao orientador a granularidade da base original que alimentou o nosso ETL.\n\n")
        
        # 1. Competitions
        comp_df = sb.competitions()
        comp_example = comp_df.iloc[0].to_dict()
        f.write("## 1. Competições (`sb.competitions()`)\n")
        f.write("Retorna quais campeonatos e temporadas estão disponíveis na base aberta. Usamos isso para mapear os IDs necessários para baixar os jogos.\n")
        f.write("```json\n")
        f.write(json.dumps(comp_example, indent=2, default=str))
        f.write("\n```\n\n")
        
        # 2. Matches
        comp_id = comp_example['competition_id']
        season_id = comp_example['season_id']
        try:
            matches_df = sb.matches(competition_id=43, season_id=106) # World Cup 2022
        except:
            matches_df = sb.matches(competition_id=comp_id, season_id=season_id)
        
        match_example = matches_df.iloc[0].to_dict()
        match_id = match_example['match_id']
        
        f.write("## 2. Partidas (`sb.matches(competition_id, season_id)`)\n")
        f.write("Retorna a lista de jogos de um campeonato específico. Contém metadados importantes de cada partida (data, placar, estádio, juiz, técnicos).\n")
        f.write("```json\n")
        f.write(json.dumps(match_example, indent=2, default=str))
        f.write("\n```\n\n")
        
        # 3. Lineups (Escalações)
        lineups = sb.lineups(match_id=match_id)
        team_name = list(lineups.keys())[0]
        lineup_df = lineups[team_name]
        lineup_example = lineup_df.iloc[0].to_dict()
        
        f.write("## 3. Escalações (`sb.lineups(match_id)`)\n")
        f.write("Retorna a ficha de cada jogador relacionado para a partida (titulares e reservas), incluindo posições nativas e país.\n")
        f.write("```json\n")
        f.write(json.dumps(lineup_example, indent=2, default=str))
        f.write("\n```\n\n")
        
        # 4. Events
        events_df = sb.events(match_id=match_id)
        # Get one specific event (e.g., a Pass)
        pass_events = events_df[events_df['type'] == 'Pass']
        if not pass_events.empty:
            event_example = pass_events.iloc[0].dropna().to_dict()
        else:
            event_example = events_df.iloc[0].dropna().to_dict()
            
        f.write("## 4. Eventos (`sb.events(match_id)`)\n")
        f.write("A base mais rica do projeto. Contém o log de tudo que aconteceu em campo ('Pass', 'Shot', 'Carry', 'Pressure', etc), junto com as coordenadas (x, y) de onde ocorreu. Foi daqui que extraímos o xG, Field Tilt e Zone 14.\n")
        f.write("```json\n")
        f.write(json.dumps(event_example, indent=2, default=str))
        f.write("\n```\n")

    print(f"Examples successfully saved to {output_md}")

if __name__ == "__main__":
    extract_examples()
