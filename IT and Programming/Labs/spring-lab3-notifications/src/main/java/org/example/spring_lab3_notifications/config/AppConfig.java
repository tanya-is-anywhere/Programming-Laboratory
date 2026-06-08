package org . example.spring_lab3_notifications . config ;
import org . springframework . context . annotation . Bean ;
import org.springframework.context.annotation.ComponentScan;
import org . springframework . context . annotation . Configuration ;
import org.springframework.context.annotation.Import;
@Configuration
@ComponentScan("org.example.controller")
public class AppConfig {

}